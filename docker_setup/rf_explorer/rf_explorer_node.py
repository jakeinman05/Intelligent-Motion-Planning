#!/usr/bin/env python3
"""Rebuilds the radio map from the rover's readings.

Every `update_period_s`:
  1. Bins the RSS readings collected so far onto the 256x256 grid.
  2. Runs the reconstruction model (CNN / GNN, or `idw`
     as a no-PyTorch baseline) to estimate the full map.
  3. Publishes the estimated map and estimated emitter location for RViz, and
     logs the error against the true map.
  4. If drive:=true, picks the next sampling spot and sends the rover there
     through ArduPilot (GUIDED mode, over MAVLink).

Publishes
  /rf/estimated_map      nav_msgs/OccupancyGrid
  /rf/estimated_emitter  geometry_msgs/PointStamped
  /rf/goal               geometry_msgs/PointStamped  (where the rover is heading)
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Float32
from geometry_msgs.msg import PointStamped

import rf_core
from ros_maps import to_occupancy_grid


def _mb_from_status(key):
    """This process's memory from /proc/self/status (VmRSS = now, VmHWM = peak), in MB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024
    return float("nan")


def container_ram_mb():
    """(used, limit) MB for the whole container (cgroup v2), i.e. the Jetson budget."""
    try:
        used = int(open("/sys/fs/cgroup/memory.current").read()) / 2**20
        lim = open("/sys/fs/cgroup/memory.max").read().strip()
        return used, (None if lim == "max" else int(lim) / 2**20)
    except OSError:
        return None, None


class Reconstructor:
    """Wraps either a trained network (via models_radiomap.py) or the IDW baseline."""
    def __init__(self, model_name, weights, models_dir, logger):
        self.name = model_name
        if model_name.lower() == "idw":
            self.net = None
            return
        import torch
        sys.path.insert(0, os.path.expanduser(models_dir))
        import models_radiomap as mr
        self.torch = torch
        torch.set_num_threads(len(os.sched_getaffinity(0)))  # the cores the container allows
        self.net = mr.build_model(model_name)
        state = torch.load(os.path.expanduser(weights), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = mr.load_state_dict_flexible(self.net, state)
        if missing or unexpected:
            logger.warn(f"Weights mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys")
        self.net.eval()
        logger.info(f"Loaded {model_name} from {weights}")

    def __call__(self, buffer, building_mask, tx_origin):
        if self.net is None:
            return rf_core.idw_reconstruct(buffer)
        x = buffer.model_input(building_mask, tx_origin)
        with self.torch.no_grad():
            out = self.net(self.torch.from_numpy(x)[None])
        if isinstance(out, (tuple, list)):
            out = out[-1]
        return rf_core.denormalise(out[0, 0].numpy().clip(0.0, 1.0))


class ArduRover:
    """Minimal MAVLink link to ArduPilot: GUIDED mode, arm, go-to in local NED."""
    def __init__(self, url, logger):
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.log = logger
        self.m = mavutil.mavlink_connection(url)
        self.log.info(f"Waiting for ArduPilot heartbeat on {url} ...")
        self.m.wait_heartbeat(timeout=30)
        self.local_ned = None

    def poll(self):
        while True:
            msg = self.m.recv_match(type="LOCAL_POSITION_NED", blocking=False)
            if msg is None:
                return
            self.local_ned = (msg.x, msg.y)

    def start(self, wp_radius_m=None):
        if wp_radius_m:
            # Rover stops within WP_RADIUS of a target (default 2 m);
            self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                      b"WP_RADIUS", float(wp_radius_m),
                                      self.mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        self.m.set_mode("GUIDED")
        self.m.arducopter_arm()    
        self.m.motors_armed_wait()
        # Ask for local position updates at 5 Hz
        self.m.mav.request_data_stream_send(
            self.m.target_system, self.m.target_component,
            self.mavutil.mavlink.MAV_DATA_STREAM_POSITION, 5, 1)
        self.log.info("Rover armed in GUIDED mode")

    def goto_ned(self, north, east):
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            self.mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b110111111000,               # use position only
            north, east, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def stop(self):
        self.m.set_mode("HOLD")


class RfExplorer(Node):
    def __init__(self):
        super().__init__("rf_explorer")
        p = lambda name, default: self.declare_parameter(name, default).value
        self.grid = rf_core.Grid(256, p("cell_size_m", 1.0))
        self.tx_power = p("tx_power_dbm", 0.0)
        self.frame = p("frame_id", "odom")
        self.drive = p("drive", False)
        self.max_step_m = p("max_step_m", 15.0)
        self.reach_m = p("reach_m", 1.5)
        self.found_m = p("found_m", 3.0)
        self.explore_weight = p("explore_weight", 0.5)
        self.max_distance_m = p("max_distance_m", 1000.0)

        scene = rf_core.load_scene(p("parquet", ""), p("row", 0))
        self.true_pl = scene["path_loss"]
        self.mask = scene.get("building_mask", np.zeros_like(self.true_pl))
        # The prior team's models were trained WITH the emitter location as input.
        # Leave this False for a fair "find the emitter" test.
        self.tx_in = scene.get("tx_origin") if p("use_tx_origin", False) else None
        if "tx_origin" in scene and scene["tx_origin"].max() > 0:
            self.true_tx = np.unravel_index(np.argmax(scene["tx_origin"]), self.true_pl.shape)
        else:
            self.true_tx = rf_core.emitter_estimate(self.true_pl)

        self.model = Reconstructor(p("model", "idw"), p("weights", ""),
                                   p("models_dir", "~/host/jetson_radiomap"), self.get_logger())
        self.buffer = rf_core.SampleBuffer(self.true_pl.shape[0])
        self.pose = None
        self.last_pose = None
        self.distance_m = 0.0
        self.goal_xy = None
        self.stall_s = p("stall_s", 10.0)
        self.progress = (time.time(), 0.0)   # (time, distance_m) when the rover last moved
        self.avoid = np.zeros_like(self.mask)  # goals the rover failed to reach
        self.settle = rf_core.SettleCheck(3, p("settle_m", 3.0) / self.grid.cell)
        self.done = False
        self.model_ms = []
        self.t0 = time.time()

        # Latched, so RViz still shows the last map/guess after the run stops
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, "/rf/estimated_map", latched)
        self.est_pub = self.create_publisher(PointStamped, "/rf/estimated_emitter", latched)
        self.goal_pub = self.create_publisher(PointStamped, "/rf/goal", latched)
        self.create_subscription(Odometry, p("odom_topic", "/odometry"), self.on_odom, 10)
        self.create_subscription(Float32, "/rf/rss_dbm", self.on_rss, 50)

        self.rover = None
        if self.drive:
            self.rover = ArduRover(p("mavlink_url", "udpin:0.0.0.0:14551"), self.get_logger())
            self.rover.start(wp_radius_m=0.8 * self.reach_m)
            self.create_timer(0.1, self.rover.poll)
        self.create_timer(p("update_period_s", 3.0), self.update)

    def on_odom(self, msg):
        pos = msg.pose.pose.position
        self.pose = (pos.x, pos.y)
        if self.last_pose is not None:
            self.distance_m += math.dist(self.pose, self.last_pose)
        self.last_pose = self.pose

    def on_rss(self, msg):
        if self.pose is None:
            return
        rc = self.grid.to_pixel(*self.pose)
        if rc is not None:
            self.buffer.add(rc[0], rc[1], self.tx_power - msg.data)

    def point(self, x, y):
        pt = PointStamped()
        pt.header.frame_id = self.frame
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x, pt.point.y = float(x), float(y)
        return pt

    def update(self):
        if self.pose is None or self.done:
            return
        t = time.perf_counter()
        pred = self.model(self.buffer, self.mask, self.tx_in)
        model_ms = (time.perf_counter() - t) * 1000
        self.map_pub.publish(to_occupancy_grid(pred, self.grid, self.frame, self.get_clock().now()))
        er, ec = rf_core.emitter_estimate(pred, self.mask)
        ex, ey = self.grid.to_world(er, ec)
        self.est_pub.publish(self.point(ex, ey))

        tx, ty = self.grid.to_world(*self.true_tx)
        rmse = float(np.sqrt(np.mean((pred - self.true_pl) ** 2)))
        est_err = math.dist((ex, ey), (tx, ty))
        to_true = math.dist(self.pose, (tx, ty))
        self.get_logger().info(
            f"samples={int(self.buffer.known.sum())} px | map RMSE={rmse:.1f} dB | "
            f"emitter guess off by {est_err:.1f} m | rover {to_true:.1f} m from emitter | "
            f"driven {self.distance_m:.1f} m")
        self.model_ms.append(model_ms)
        used, lim = container_ram_mb()
        box = "" if used is None else f" | container RAM {used:.0f}" + ("" if lim is None else f"/{lim:.0f}") + " MB"
        self.get_logger().info(
            f"  compute: {self.model.name} {model_ms:.0f} ms | explorer RAM "
            f"{_mb_from_status('VmRSS'):.0f} MB (peak {_mb_from_status('VmHWM'):.0f}){box}")

        if to_true <= self.found_m:
            self.get_logger().info(
                f"FOUND: rover within {self.found_m} m of the emitter after "
                f"{self.distance_m:.1f} m driven, {time.time() - self.t0:.0f} s")
            self.finish()
            return
        if self.distance_m > self.max_distance_m:
            self.get_logger().info("Gave up: max_distance_m reached")
            self.finish()
            return
        settled = self.settle.update((er, ec))
        if self.drive:
            self.steer(pred, (er, ec) if settled else None)

    def steer(self, pred, home_rc=None):
        now = time.time()
        if self.distance_m - self.progress[1] > 0.2:
            self.progress = (now, self.distance_m)
        if self.goal_xy is not None and math.dist(self.pose, self.goal_xy) > self.reach_m:
            if now - self.progress[0] < self.stall_s:
                return  # still on the way
            # Not moving (blocked by a wall, or ArduPilot thinks it has arrived): give up on it
            self.get_logger().warn(f"Stuck for {self.stall_s:.0f} s; picking a new goal")
            g = self.grid.to_pixel(*self.goal_xy)
            if g is not None:
                r = max(1, int(1.0 / self.grid.cell))
                self.avoid[max(0, g[0] - r):g[0] + r + 1, max(0, g[1] - r):g[1] + r + 1] = 1
            self.progress = (now, self.distance_m)
        rover_rc = self.grid.to_pixel(*self.pose)
        if rover_rc is None:
            self.get_logger().warn("Rover is outside the radio map; not steering")
            return
        step_px = int(self.max_step_m / self.grid.cell)
        blocked = np.maximum(self.mask, self.avoid)
        goal = None
        if home_rc is not None:  # estimate has stopped moving: drive at it
            goal = rf_core.homing_goal(rover_rc, home_rc, step_px, blocked)
            if goal is not None and math.dist(goal, rover_rc) * self.grid.cell < self.reach_m:
                goal = None  # already there; explore around it instead
        if goal is None:
            goal = rf_core.choose_goal(pred, self.buffer, blocked, rover_rc,
                                       max_step_px=step_px,
                                       explore_weight=self.explore_weight)
        if goal is None or self.rover.local_ned is None:
            return
        gx, gy = self.grid.to_world(*goal)
        self.goal_xy = (gx, gy)
        self.progress = (time.time(), self.distance_m)
        # Gazebo world is ENU (x east, y north); ArduPilot local frame is NED.
        dx, dy = gx - self.pose[0], gy - self.pose[1]
        n0, e0 = self.rover.local_ned
        self.rover.goto_ned(n0 + dy, e0 + dx)
        self.goal_pub.publish(self.point(gx, gy))

    def finish(self):
        self.done = True
        if self.model_ms:
            used, lim = container_ram_mb()
            self.get_logger().info(
                f"SUMMARY {self.model.name}: driven {self.distance_m:.1f} m, "
                f"{time.time() - self.t0:.0f} s, model avg {np.mean(self.model_ms):.0f} ms "
                f"(max {np.max(self.model_ms):.0f}), explorer peak RAM {_mb_from_status('VmHWM'):.0f} MB"
                + ("" if lim is None else f", container limit {lim:.0f} MB"))
        if self.rover:
            self.rover.stop()


def main():
    rclpy.init()
    rclpy.spin(RfExplorer())


if __name__ == "__main__":
    main()