#!/usr/bin/env python3
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray


class ArucoDetector(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        self.marker_size = self.declare_parameter('marker_size', 0.07).value
        self.camera_frame = self.declare_parameter('camera_frame', 'camera_rgb_optical_frame').value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.max_distance = self.declare_parameter('max_distance', 2.5).value
        self.min_observations = self.declare_parameter('min_observations', 3).value
        self.process_rate_hz = self.declare_parameter('process_rate_hz', 10.0).value

        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.detector_params = cv2.aruco.DetectorParameters_create()

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.last_processed = None

        self.observations: dict[int, list[tuple[float, float, float]]] = {}

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, qos_profile_sensor_data)
        self.markers_pub = self.create_publisher(MarkerArray, '/aruco/markers', 10)
        self.debug_pub = self.create_publisher(Image, '/aruco/debug_image', 1)

        self.get_logger().info('aruco det redy')


    def camera_info_cb(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def image_cb(self, msg: Image):
        if self.camera_matrix is None:
            return
        now = self.get_clock().now()
        if self.last_processed is not None and \
                (now - self.last_processed).nanoseconds < 1e9 / self.process_rate_hz:
            return
        self.last_processed = now

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        corners, ids = self.detect_markers(frame)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for marker_corners, marker_id in zip(corners, ids.ravel()):
                tvec = self.estimate_pose(marker_corners)
                distance = float(np.linalg.norm(tvec))

                if distance > self.max_distance:
                    continue

                point_map = self.camera_point_to_map(tvec, msg.header.stamp)
                if point_map is None:
                    continue
                self.add_observation(int(marker_id), point_map)
                self.get_logger().info(
                    f'ID {marker_id}: {distance:.2f} from cam, '
                    f'map=({point_map[0]:.2f}, {point_map[1]:.2f})',
                    throttle_duration_sec=1.0)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
        self.publish_markers()

    def detect_markers(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(frame_gray, self.dictionary, parameters = self.detector_params)
        return corners, ids

    def estimate_pose(self, marker_corners) -> np.ndarray:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [marker_corners], self.marker_size, self.camera_matrix, self.dist_coeffs)

        return tvecs[0].ravel()

    def camera_point_to_map(self, tvec: np.ndarray, stamp):
        popo = PointStamped()
        popo.header.frame_id = self.camera_frame
        popo.header.stamp = stamp
        popo.point.x = float(tvec[0])
        popo.point.y = float(tvec[1])
        popo.point.z = float(tvec[2])

        try:
            p_map = self.tf_buffer.transform(popo, self.map_frame, timeout=Duration(seconds=0.2))
            return (p_map.point.x, p_map.point.y, p_map.point.z)
        except tf2_ros.TransformException:
            return None


    def add_observation(self, marker_id: int, point_map):
        self.observations.setdefault(marker_id, []).append(point_map)
        pass

    def get_estimates(self) -> dict[int, tuple[float, float, float]]:

        estimates = {}
        for marker_id, points in self.observations.items():
            if len(points) < self.min_observations:
                continue
            estimates[marker_id] = tuple(np.mean(points, axis=0))
        return estimates

    def publish_markers(self):
        array = MarkerArray()
        for marker_id, (x, y, z) in sorted(self.get_estimates().items()):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'aruco'
            m.id = marker_id
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.09
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
            array.markers.append(m)
        self.markers_pub.publish(array)


def main():
    rclpy.init()
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
