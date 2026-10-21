"""Shared map maths for the RF exploration sim (numpy + scipy, no ROS, no torch).

Grid convention used by every node: a 256x256 map is centred on world (0, 0) by
default. Pixel (row r, col c) covers world x = origin_x + (c + 0.5) * cell,
y = origin_y + (r + 0.5) * cell. Sensor, rebuilder and explorer all use this one
mapping, so they always agree with each other.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

MIN_DB, MAX_DB = 30.0, 140.0   # path-loss normalisation used in training
NOISE_FLOOR_DBM = -120.0


class Grid:
    def __init__(self, n=256, cell_size_m=1.0, origin_x=None, origin_y=None):
        self.n, self.cell = n, cell_size_m
        half = n * cell_size_m / 2.0
        self.ox = -half if origin_x is None else origin_x
        self.oy = -half if origin_y is None else origin_y

    def to_pixel(self, x, y):
        c = int(np.floor((x - self.ox) / self.cell))
        r = int(np.floor((y - self.oy) / self.cell))
        if 0 <= r < self.n and 0 <= c < self.n:
            return r, c
        return None

    def to_world(self, r, c):
        return self.ox + (c + 0.5) * self.cell, self.oy + (r + 0.5) * self.cell


def load_scene(path, row=0):
    """Read one scene from the team's parquet format -> dict of 2D arrays."""
    df = pd.read_parquet(path)
    rec = df.iloc[row]
    n = int(round(np.sqrt(len(rec["path_loss"]))))
    out = {k: np.asarray(rec[k], dtype=np.float32).reshape(n, n)
           for k in ("building_mask", "tx_origin", "path_loss") if k in df.columns}
    return out


def normalise(path_loss_db):
    return np.clip((path_loss_db - MIN_DB) / (MAX_DB - MIN_DB), 0.0, 1.0)


def denormalise(x):
    return x * (MAX_DB - MIN_DB) + MIN_DB


class SampleBuffer:
    """Accumulates sparse path-loss readings on the grid (mean per pixel)."""
    def __init__(self, n=256):
        self.sum = np.zeros((n, n), np.float64)
        self.count = np.zeros((n, n), np.int32)

    def add(self, r, c, path_loss_db):
        self.sum[r, c] += path_loss_db
        self.count[r, c] += 1

    @property
    def known(self):
        return self.count > 0

    def mean_db(self):
        out = np.zeros_like(self.sum)
        k = self.known
        out[k] = self.sum[k] / self.count[k]
        return out

    def model_input(self, building_mask, tx_origin=None):
        """(3, n, n) float32 tensor layout the models were trained on:
        [building_mask, tx_origin, sparse_path_loss_norm]. Unknown pixels are 0."""
        sparse = np.zeros(self.sum.shape, np.float32)
        k = self.known
        # Keep known pixels strictly > 0, since PartialConvMAE treats 0 as "unknown"
        sparse[k] = np.maximum(normalise(self.mean_db()[k]), 1e-3)
        tx = np.zeros_like(sparse) if tx_origin is None else tx_origin.astype(np.float32)
        return np.stack([building_mask.astype(np.float32), tx, sparse])


def idw_reconstruct(buffer, power=2.0, k=16):
    """Baseline without a neural net: inverse-distance-weighted interpolation (dB)
    over the k nearest sampled pixels."""
    n = buffer.sum.shape[0]
    rs, cs = np.nonzero(buffer.known)
    if len(rs) == 0:
        return np.full((n, n), (MIN_DB + MAX_DB) / 2.0)
    vals = buffer.mean_db()[rs, cs]
    k = min(k, len(rs))
    yy, xx = np.mgrid[0:n, 0:n]
    d, i = cKDTree(np.c_[rs, cs]).query(np.c_[yy.ravel(), xx.ravel()], k=k)
    d, i = d.reshape(n * n, k), i.reshape(n * n, k)
    w = 1.0 / (d ** 2 + 1.0) ** (power / 2.0)
    return ((w * vals[i]).sum(-1) / w.sum(-1)).reshape(n, n)


def emitter_estimate(path_loss_db, building_mask=None):
    """Pixel with the lowest predicted path loss (strongest signal), off-building."""
    pl = path_loss_db.copy()
    if building_mask is not None:
        pl[building_mask > 0.5] = np.inf
    return np.unravel_index(np.argmin(pl), pl.shape)


def choose_goal(pred_db, buffer, building_mask, rover_rc, max_step_px=40,
                explore_weight=0.5, min_step_px=3):
    """Next pixel to drive to: strong predicted signal + far from past samples.

    Score is the signal term (0..1, higher = stronger) plus explore_weight times
    novelty (0..1), limited to free pixels within max_step_px of the rover.
    """
    n = pred_db.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    r0, c0 = rover_rc
    step = np.hypot(yy - r0, xx - c0)
    signal = 1.0 - normalise(pred_db)
    if buffer.known.any():
        dmin = distance_transform_edt(~buffer.known)   # px to nearest sample
        novelty = np.clip(dmin / max_step_px, 0.0, 1.0)
    else:
        novelty = np.ones((n, n))
    score = signal + explore_weight * novelty
    allowed = (step <= max_step_px) & (step >= min_step_px)
    if building_mask is not None:
        allowed &= building_mask < 0.5
    score[~allowed] = -np.inf
    if not np.isfinite(score).any():
        return None
    return np.unravel_index(np.argmax(score), score.shape)


def homing_goal(rover_rc, target_rc, max_step_px, building_mask=None):
    """Step straight toward target_rc, at most max_step_px. None if that lands on a building."""
    r0, c0 = rover_rc
    dr, dc = target_rc[0] - r0, target_rc[1] - c0
    d = float(np.hypot(dr, dc))
    s = 1.0 if d <= max_step_px else (max_step_px - 0.75) / d  # rounding stays inside the limit
    r, c = int(round(r0 + s * dr)), int(round(c0 + s * dc))
    if building_mask is not None and building_mask[r, c] > 0.5:
        return None
    return r, c


class SettleCheck:
    """True once the last `window` emitter estimates all lie within tol_px of each other."""
    def __init__(self, window=3, tol_px=5.0):
        self.window, self.tol, self.hist = window, tol_px, []

    def update(self, rc):
        self.hist = (self.hist + [rc])[-self.window:]
        if len(self.hist) < self.window:
            return False
        pts = np.array(self.hist, float)
        return float(np.max(np.hypot(*(pts - pts.mean(0)).T))) <= self.tol
