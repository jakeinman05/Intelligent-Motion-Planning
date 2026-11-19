import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import math
import random

class RFMeasurementNode(Node):
    def __init__(self):
        super().__init__('rf_measurement')

        self.publisher = self.create_publisher(PointStamped, "rf/samples", 100)
        self.timer = self.create_timer(1.0, self.publish_measurement)
        self.sample_index = 0

    def publish_measurement(self):
        msg = PointStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        i = self.sample_index

        msg.point.x = float(2 + (i % 12)  * 4)
        msg.point.y = float(2 + ((i // 12) % 12) * 4)

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = RFMeasurementNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()