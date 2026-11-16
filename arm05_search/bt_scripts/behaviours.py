import math

import py_trees
import py_trees_ros
import geometry_msgs.msg as geometry_msgs
from nav2_msgs.action import NavigateToPose


def make_twist(node, linear_x: float = 0.0, angular_z: float = 0.0) -> geometry_msgs.TwistStamped:
    msg = geometry_msgs.TwistStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.twist.linear.x = linear_x
    msg.twist.angular.z = angular_z
    return msg


class CmdVelBehaviour(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, topic_name: str = "/cmd_vel"):
        super().__init__(name=name)
        self.topic_name = topic_name
        self.node = None
        self.publisher = None

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
        except KeyError as e:
            raise KeyError(f"didn't find 'node' in setup's kwargs [{self.qualified_name}]") from e
        self.publisher = self.node.create_publisher(
            msg_type=geometry_msgs.TwistStamped,
            topic=self.topic_name,
            qos_profile=py_trees_ros.utilities.qos_profile_latched(),
        )

    def publish(self, linear_x: float = 0.0, angular_z: float = 0.0):
        self.publisher.publish(make_twist(self.node, linear_x, angular_z))

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID and self.publisher is not None:
            self.publish(0.0, 0.0)


class Stop(CmdVelBehaviour):
    def update(self) -> py_trees.common.Status:

        self.publish(0.0, 0.0)
        return py_trees.common.Status.SUCCESS


class ScanAround(CmdVelBehaviour):
    def __init__(self, name: str, angle: float = 2 * math.pi, vel: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.angle = angle
        self.vel = vel
        self.start_time = None

    def initialise(self) -> None:
        self.start_time = self.node.get_clock().now()

    def update(self) -> py_trees.common.Status:
        elapsed = (self.node.get_clock().now() - self.start_time).nanoseconds * 1e-9
        self.feedback_message = f"elapsed {elapsed:.1f}s"

        duration = self.angle / self.vel
        if elapsed < duration:
            angular_z = self.vel
            self.publish(0.0, angular_z)
            return py_trees.common.Status.RUNNING
        self.publish(0.0, 0.0)
        return py_trees.common.Status.SUCCESS


class NextWaypoint(py_trees.behaviour.Behaviour):

    def __init__(self, name: str, waypoints: list[tuple[float, float]], frame_id: str = "map"):
        super().__init__(name=name)
        self.waypoints = list(waypoints)
        self.frame_id = frame_id
        self.index = 0
        self.blackboard = self.attach_blackboard_client(name=name)
        self.blackboard.register_key(key="nav_goal", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        if self.index >= len(self.waypoints):
            return py_trees.common.Status.FAILURE
        x, y = self.waypoints[self.index]
        self.index += 1
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.frame_id
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0
        self.blackboard.nav_goal = goal
        self.feedback_message = f"punkt {self.index}/{len(self.waypoints)}"
        return py_trees.common.Status.SUCCESS
