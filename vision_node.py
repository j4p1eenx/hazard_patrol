#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO


class YOLOProcessorNode(Node):
    def __init__(self):
        super().__init__('yolo_processor_node')

        self.bridge = CvBridge()

        # Load the lightweight YOLOv8 nano model (downloads on first run)
        self.get_logger().info('Loading YOLOv8n model...')
        self.model = YOLO('yolov8n.pt')
        self.get_logger().info('YOLOv8n model loaded successfully!')

        # Sensor QoS (best effort, small queue): drops old frames instead of
        # building up a backlog while YOLO is busy.
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data,
        )

        # Annotated image (view with rqt_image_view)
        self.publisher = self.create_publisher(Image, '/camera/detected_objects', 10)

        self.get_logger().info('YOLO Vision Node has been started.')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e!r}')
            return

        # Run YOLO inference (conf=0.5 sets a 50% confidence threshold)
        results = self.model(cv_image, conf=0.5, verbose=False)

        # Draw bounding boxes and labels (returns a BGR uint8 numpy array)
        annotated_frame = results[0].plot()

        # Log detected objects
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            self.get_logger().info(
                f'Detected: {class_name} ({confidence:.2f})',
                throttle_duration_sec=1.0,
            )

        try:
            # Build the Image message by hand instead of using
            # cv_bridge.cv2_to_imgmsg(), which is what was failing.
            frame = np.ascontiguousarray(annotated_frame, dtype=np.uint8)
            out = Image()
            out.header = msg.header
            out.height = int(frame.shape[0])
            out.width = int(frame.shape[1])
            out.encoding = 'bgr8'
            out.is_bigendian = 0
            out.step = out.width * 3
            out.data = frame.tobytes()
            self.publisher.publish(out)
        except Exception as e:
            self.get_logger().error(f'Failed to publish annotated image: {e!r}')


def main(args=None):
    rclpy.init(args=args)
    node = YOLOProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
