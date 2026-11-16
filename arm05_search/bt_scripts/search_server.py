#!/usr/bin/env python3

import sys
import time

import py_trees
import py_trees_ros
import py_trees.console as console
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from py_trees_ros import subscribers as py_trees_subscribers
import visualization_msgs.msg as visualization_msgs
from geometry_msgs.msg import Point
from nav2_msgs.action import NavigateToPose

from arm05_search_interfaces.action import FindMarkers

try:
    from bt_scripts import behaviours
except ImportError:
    import behaviours

TICK_RATE_HZ = 10.0

WAYPOINTS = [
    (-1.0, 0.5), (-1.0, 3.0), (-5.0, 4.5), (-5.0, 0.5), (-5.0, -4.5),     # lewe skrzydło
    (-1.0, -5.0), (2.5, -6.2), (4.0, -2.5),                                # dół, środek
    (4.0, 3.5), (8.5, 3.5), (12.0, 4.0),                                   # górne pokoje
    (10.0, -1.5), (16.2, 2.0), (16.2, -1.7), (10.5, -5.7),                 # prawa strona
]



def all_markers_found(blackboard) -> bool:
    found_ids = {marker.id for marker in blackboard.markers}
    return len(found_ids) >= blackboard.markers_to_find



def create_root() -> py_trees.behaviour.Behaviour:
    root = py_trees.composites.Parallel(
        name="Root",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False),
    )

    markers2bb = py_trees_subscribers.ToBlackboard(
        name="Markers2BB",
        topic_name="/aruco/markers",
        topic_type=visualization_msgs.MarkerArray,
        blackboard_variables={'markers': 'markers'},
        initialise_variables={'markers': []},
        qos_profile=py_trees_ros.utilities.qos_profile_unlatched(),
        clearing_policy=py_trees.common.ClearingPolicy.NEVER,
    )

    stop = behaviours.Stop(name="Stop")
    next_waypoint = behaviours.NextWaypoint(name="NextWaypoint", waypoints=WAYPOINTS)
    go_to = py_trees_ros.action_clients.FromBlackboard(
        name="GoTo",
        action_type=NavigateToPose,
        action_name="navigate_to_pose",
        key="nav_goal",
    )
    scan_around = behaviours.ScanAround(name="ScanAround", vel=1.2)

    all_found = py_trees.decorators.EternalGuard(
        name="AllFound?",
        child=stop,
        condition=all_markers_found,
        blackboard_keys={'markers', 'markers_to_find'},
    )
    go_or_skip = py_trees.decorators.FailureIsSuccess(
        name="GoToOrSkip",
        child=go_to,
    )
    patrol = py_trees.composites.Sequence(
        name="Patrol",
        memory=True,
        children=[next_waypoint, go_or_skip, scan_around],
    )
    mission = py_trees.composites.Selector(
        name="Mission",
        memory=False,
        children=[all_found, patrol],
    )

    root.add_children([markers2bb, mission])
    return root



class SearchServer(Node):

    def __init__(self):
        super().__init__('search_server')
        self.busy = False

        self.tree = py_trees_ros.trees.BehaviourTree(root=create_root(), unicode_tree_debug=False)
        self.tree.setup(node=self, timeout=15.0)

        self.bb = py_trees.blackboard.Client(name="SearchServer")
        self.bb.register_key(key="markers_to_find", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="markers", access=py_trees.common.Access.READ)
        self.bb.markers_to_find = 5

        self.action_server = ActionServer(
            self, FindMarkers, 'find_markers',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        print(py_trees.display.unicode_tree(self.tree.root))
        self.get_logger().info('search_server rdy')

    def goal_callback(self, goal_request):
        if self.busy:
            self.get_logger().warn('goal declined')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.busy = True
        goal = goal_handle.request
        self.bb.markers_to_find = goal.markers_to_find
        self.get_logger().info(f'looking for {goal.markers_to_find} boxes, limit {goal.timeout_s:.0f} s')

        start = time.monotonic()
        feedback = FindMarkers.Feedback()
        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self.stop_tree()
                    goal_handle.canceled()
                    self.get_logger().info('cancelled')
                    return self.make_result()

                self.tree.tick()

                elapsed = time.monotonic() - start
                if all_markers_found(self.bb):
                    self.get_logger().info('all found')
                    break
                if self.tree.root.status == py_trees.common.Status.FAILURE:
                    self.get_logger().warn('end of the route, not all found')
                    break
                if elapsed > goal.timeout_s:
                    self.get_logger().warn('time limit exceeded, not all found')
                    break

                tip = self.tree.tip()
                feedback.markers_found = len({m.id for m in self.bb.markers})
                feedback.elapsed_s = float(elapsed)
                feedback.tree_tip = f'{tip.name}: {tip.feedback_message}' if tip is not None else ''
                goal_handle.publish_feedback(feedback)

                time.sleep(1.0 / TICK_RATE_HZ)
        finally:
            self.stop_tree()
            self.busy = False

        goal_handle.succeed()
        return self.make_result()

    def stop_tree(self):

        self.tree.root.stop(py_trees.common.Status.INVALID)

    def make_result(self) -> FindMarkers.Result:
        result = FindMarkers.Result()
        for m in sorted(self.bb.markers, key=lambda m: m.id):
            result.ids.append(m.id)
            result.positions.append(Point(x=m.pose.position.x, y=m.pose.position.y, z=m.pose.position.z))
        result.success = len(set(result.ids)) >= self.bb.markers_to_find
        self.get_logger().info(f'Wynik: {list(zip(result.ids, [(round(p.x, 2), round(p.y, 2)) for p in result.positions]))}')
        return result


def main():
    if '--render' in sys.argv:
        py_trees.display.render_dot_tree(create_root(), name="search_tree")
        return

    rclpy.init()
    try:
        server = SearchServer()
    except py_trees_ros.exceptions.TimedOutError as e:
        console.logerror(console.red + f"failed to setup the tree, aborting [{e}]" + console.reset)
        rclpy.try_shutdown()
        sys.exit(1)

    executor = MultiThreadedExecutor()
    executor.add_node(server)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        server.tree.shutdown()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
