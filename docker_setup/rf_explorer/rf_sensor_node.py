#!/usr/bin/env python3
"""Simulated RF sensor: reads the Sionna radio map at the rover's position.

Publishes
  /rf/rss_dbm      std_msgs/Float32         received power = tx_power - path_loss + noise
  /rf/true_map     nav_msgs/OccupancyGrid   ground-truth map for RViz (100 = strongest)
  /rf/true_emitter geometry_msgs/PointStamped  where the emitter really is
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Float32
from geometry_msgs.msg import PointStamped
from scipy.interpolate import RegularGridInterpolator

import rf_core
from ros_maps import to_occupancy_grid


class RfSensor(Node):
    def __init__(self):
        super().__init__("rf_sensor")
        p = lambda name, default: self.declare_parameter(name, default).value
        path = p("parquet", "")
        row = p("row", 0)
        self.grid = rf_core.Grid(256, p("cell_size_m", 1.0))
        self.tx_power = p("tx_power_dbm", 0.0)
        self.noise_db = p("noise_std_db", 2.0)
        self.frame = p("frame_id", "odom")

        scene = rf_core.load_scene(path, row)
        self.true_pl = scene["path_loss"]
        n = self.true_pl.shape[0]
        g = self.grid
        xs = g.ox + (np.arange(n) + 0.5) * g.cell
        ys = g.oy + (np.arange(n) + 0.5) * g.cell
        self.lookup = RegularGridInterpolator((ys, xs), self.true_pl.astype(float),
                                              bounds_error=False, fill_value=rf_core.MAX_DB)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(Float32, "/rf/rss_dbm", 10)
        map_pub = self.create_publisher(OccupancyGrid, "/rf/true_map", latched)
        tx_pub = self.create_publisher(PointStamped, "/rf/true_emitter", latched)
        map_pub.publish(to_occupancy_grid(self.true_pl, self.grid, self.frame, self.get_clock().now()))
        if "tx_origin" in scene and scene["tx_origin"].max() > 0:
            r, c = np.unravel_index(np.argmax(scene["tx_origin"]), scene["tx_origin"].shape)
        else:
            r, c = rf_core.emitter_estimate(self.true_pl)
        x, y = self.grid.to_world(r, c)
        pt = PointStamped()
        pt.header.frame_id = self.frame
        pt.point.x, pt.point.y = float(x), float(y)
        tx_pub.publish(pt)
        self.get_logger().info(
            f"Scene {row} of {path}: map {n * self.grid.cell:.0f} m wide, "
            f"true emitter at ({x:.1f}, {y:.1f}) m")
        self.create_subscription(Odometry, p("odom_topic", "/odometry"), self.on_odom, 10)

    def on_odom(self, msg):
        pos = msg.pose.pose.position
        pl = float(self.lookup([[pos.y, pos.x]])[0])
        rss = self.tx_power - pl + np.random.normal(0.0, self.noise_db)
        self.pub.publish(Float32(data=float(rss)))


def main():
    rclpy.init()
    rclpy.spin(RfSensor())


if __name__ == "__main__":
    main()
