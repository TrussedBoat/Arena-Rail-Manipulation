"""Convert semantic detections into RViz sphere and text markers."""

import rclpy
import json
from pathlib import Path
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from interface.msg import DetectedObjectArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class SemanticRvizVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("semantic_rviz_visualizer")
        self.declare_parameter("input_topic", "/semantic/objects")
        self.declare_parameter("marker_topic", "/semantic/rviz/markers")
        self.declare_parameter("confirmed_only", False)
        # A zero lifetime is infinite. The registry publishes only when it
        # changes, so persistent markers are needed for a stable RViz map.
        self.declare_parameter("marker_lifetime_sec", 0.0)
        lifetime = float(self.get_parameter("marker_lifetime_sec").value)
        self._lifetime = Duration(
            sec=int(lifetime), nanosec=int((lifetime % 1.0) * 1_000_000_000)
        )
        self._publisher = self.create_publisher(
            MarkerArray, str(self.get_parameter("marker_topic").value), 10
        )
        registry_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._subscription = self.create_subscription(
            DetectedObjectArray,
            str(self.get_parameter("input_topic").value),
            self._callback,
            registry_qos,
        )
        self.get_logger().info(
            "RViz markers: %s -> %s"
            % (
                self.get_parameter("input_topic").value,
                self.get_parameter("marker_topic").value,
            )
        )

    def _callback(self, message: DetectedObjectArray) -> None:
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        frame_id = message.header.frame_id or "global_origin"
        objects = message.objects
        if bool(self.get_parameter("confirmed_only").value):
            objects = [item for item in objects if item.state == "confirmed"]
            
        point_clouds = {}
        try:
            json_path = Path("semantic_objects.json")
            if json_path.exists():
                with json_path.open() as f:
                    payload = json.load(f)
                    if isinstance(payload, dict) and isinstance(payload.get("objects"), list):
                        for obj in payload["objects"]:
                            if "id" in obj and "point_cloud" in obj:
                                point_clouds[obj["id"]] = obj["point_cloud"]
        except Exception as e:
            self.get_logger().debug(f"Failed to load point clouds: {e}")

        for index, item in enumerate(objects):
            marker_id = index * 2
            radius = max(0.035, min(0.12, float(item.position_stddev_m) * 2.0))
            sphere = Marker()
            sphere.header = message.header
            sphere.header.frame_id = frame_id
            sphere.ns = "semantic_objects"
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = item.position
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = radius * 2.0
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = (0.1, 0.85, 0.25, 0.85)
            sphere.lifetime = self._lifetime
            markers.markers.append(sphere)

            label = Marker()
            label.header = sphere.header
            label.ns = "semantic_labels"
            label.id = marker_id + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = item.position
            label.pose.position.z += radius + 0.015
            label.pose.orientation.w = 1.0
            label.scale.z = 0.08
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f"{item.class_name}[{item.state}]-{item.confidence:.2f}"
            label.lifetime = self._lifetime
            markers.markers.append(label)

            if item.object_id in point_clouds and point_clouds[item.object_id]:
                cloud_marker = Marker()
                cloud_marker.header = sphere.header
                cloud_marker.ns = "semantic_point_cloud"
                cloud_marker.id = marker_id + 2
                cloud_marker.type = Marker.POINTS
                cloud_marker.action = Marker.ADD
                cloud_marker.pose.orientation.w = 1.0
                cloud_marker.scale.x = 0.015
                cloud_marker.scale.y = 0.015
                cloud_marker.lifetime = self._lifetime
                
                for pt in point_clouds[item.object_id]:
                    p = Point()
                    p.x, p.y, p.z = float(pt["x"]), float(pt["y"]), float(pt["z"])
                    c = ColorRGBA()
                    c.r, c.g, c.b, c.a = float(pt["r"])/255.0, float(pt["g"])/255.0, float(pt["b"])/255.0, 1.0
                    cloud_marker.points.append(p)
                    cloud_marker.colors.append(c)
                
                markers.markers.append(cloud_marker)
        self._publisher.publish(markers)


def main() -> None:
    rclpy.init()
    node = SemanticRvizVisualizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
