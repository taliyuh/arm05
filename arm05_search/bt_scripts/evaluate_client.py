#!/usr/bin/env python3
import re
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from arm05_search_interfaces.action import FindMarkers

GROUND_TRUTH = Path.home() / '.ros' / 'arm05_sim' / 'logs' / 'latest.txt'


def read_ground_truth(path: Path) -> dict[int, tuple[float, float]]:
    truth = {}
    for line in path.read_text().splitlines():
        m = re.match(r'Aruco (\d+): x: (\S+) y: (\S+)', line)
        if m:
            truth[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return truth


def print_report(result: FindMarkers.Result, truth: dict[int, tuple[float, float]]):
    print(f'\nsuccess = {result.success}, znaleziono {len(result.ids)} z {len(truth)}')
    print(f'{"ID":>3} {"wynik (x, y)":>18} {"prawda (x, y)":>18} {"błąd [m]":>9}')
    errors = []
    for marker_id, p in zip(result.ids, result.positions):
        if marker_id not in truth:
            print(f'{marker_id:>3}  ({p.x:6.2f}, {p.y:6.2f})   not in latest.txt')
            continue
        tx, ty = truth[marker_id]

        error = np.hypot(p.x - tx, p.y - ty)
        errors.append(error)
        print(f'{marker_id:>3}  ({p.x:6.2f}, {p.y:6.2f})   ({tx:6.2f}, {ty:6.2f})  {error:9.3f}')
    missing = sorted(set(truth) - set(result.ids))
    if missing:
        print(f'not found: {missing}')
    if errors:
        print(f'avg error: {np.nanmean(errors):.3f} m, max: {np.nanmax(errors):.3f} m')


class EvaluateClient(Node):

    def __init__(self):
        super().__init__('evaluate_client')
        self.client = ActionClient(self, FindMarkers, 'find_markers')

    def run(self, markers_to_find: int, timeout_s: float):
        self.get_logger().info('waitin for /find_markers')
        self.client.wait_for_server()
        goal = FindMarkers.Goal(markers_to_find=markers_to_find, timeout_s=timeout_s)
        send_future = self.client.send_goal_async(goal, feedback_callback=self.feedback_cb)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('goal rejected')
            return None
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result().result

    def feedback_cb(self, msg):
        fb = msg.feedback
        self.get_logger().info(
            f'[{fb.elapsed_s:6.1f} s] found {fb.markers_found}, active: {fb.tree_tip}',
            throttle_duration_sec=2.0)


def main():
    markers_to_find = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    timeout_s = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    rclpy.init()
    node = EvaluateClient()
    try:
        result = node.run(markers_to_find, timeout_s)
        if result is not None:
            print_report(result, read_ground_truth(GROUND_TRUTH))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
