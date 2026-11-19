import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import math
import random

class RFMeasurementNode(Node):
    def __init__(self):
        super().__init__('rf_measurement')

        self.publisher = self.create_publisher(PointStamped, "rf/samples", 100)
        self.timer = self.create_timer(0.001, self.publish_measurement)

    def publish_measurement(self):
        msg = PointStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.point.x = random.uniform(0.0, 900.0)
        msg.point.y = random.uniform(0.0, 900.0)

        emitter_x = 450.0
        emitter_y = 450.0

        distance = math.hypot(msg.point.x - emitter_x, msg.point.y - emitter_y)

        msg.point.z = (45.0 + 20.0 * math.log10(max(distance, 1.0)) + random.gauss(0.0, 2.0))

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