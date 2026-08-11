"""Publish wrist-camera images annotated with the raw YOLO detections."""

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .detector import create_detector


class YoloDebugNode(Node):
    """Run the configured detector on RGB frames and publish labelled boxes."""

    def __init__(self) -> None:
        super().__init__("yolo_debug")
        self.declare_parameter(
            "color_topic", "/sim/rail_franka1/cam/wrist/color/image_raw"
        )
        self.declare_parameter("output_topic", "/semantic/debug/yolo")
        self.declare_parameter("model_path", "")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("image_size", 640)
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("max_detections", 100)

        self._bridge = CvBridge()
        self._detector = create_detector(
            backend="ultralytics",
            model_path=str(self.get_parameter("model_path").value),
            device=str(self.get_parameter("device").value),
            image_size=int(self.get_parameter("image_size").value),
            confidence_threshold=float(
                self.get_parameter("confidence_threshold").value
            ),
            max_detections=int(self.get_parameter("max_detections").value),
        )
        self._publisher = self.create_publisher(
            Image, str(self.get_parameter("output_topic").value), 10
        )
        self._subscription = self.create_subscription(
            Image,
            str(self.get_parameter("color_topic").value),
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "YOLO debug ready: model=%s, input=%s, output=%s"
            % (
                self._detector.model_id,
                self.get_parameter("color_topic").value,
                self.get_parameter("output_topic").value,
            )
        )

    def _image_callback(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            detections = self._detector.detect(frame)
        except (CvBridgeError, RuntimeError, ValueError) as exc:
            self.get_logger().warning(f"YOLO debug frame skipped: {exc}")
            return

        for detection in detections:
            left, top, right, bottom = (round(v) for v in detection.bbox_xyxy)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{detection.class_name} {detection.confidence:.2f}"
            label_y = max(top - 8, 20)
            cv2.putText(
                frame,
                label,
                (left, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"YOLO: {len(detections)} detection(s)",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        output = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        output.header = message.header
        self._publisher.publish(output)


def main() -> None:
    rclpy.init()
    node = YoloDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
