"""Small ROS helpers shared by the RF nodes."""
import numpy as np
from nav_msgs.msg import OccupancyGrid

import rf_core


def to_occupancy_grid(path_loss_db, grid, frame_id, stamp):
    """Path-loss map -> OccupancyGrid for RViz. 100 = strongest signal, 0 = weakest."""
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp.to_msg()
    msg.info.resolution = float(grid.cell)
    msg.info.width = msg.info.height = int(grid.n)
    msg.info.origin.position.x = float(grid.ox)
    msg.info.origin.position.y = float(grid.oy)
    msg.info.origin.orientation.w = 1.0
    strength = 100.0 * (1.0 - rf_core.normalise(np.asarray(path_loss_db)))
    msg.data = np.clip(np.round(strength), 0, 100).astype(np.int8).ravel().tolist()
    return msg
