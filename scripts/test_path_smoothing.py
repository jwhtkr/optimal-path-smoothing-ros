#! /usr/bin/env python

import time

import matplotlib.pyplot as plt

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import rospy

from path_smoothing.srv import SmoothPath


if __name__ == "__main__":
    test_path = Path()
    test_path.header.stamp = rospy.Time()

    pose_stamped = PoseStamped()
    pose_stamped.header.stamp = test_path.header.stamp
    pose_stamped.pose.position.x = 0
    pose_stamped.pose.position.y = 0
    test_path.poses = [pose_stamped]

    durations = [5, 6, 10, 14, 16, 21, 26, 30, 34]
    points = [
        (3, 3),
        (5, 2),
        (5, -2),
        (7, -2),
        (3, 0),
        (-1, -2),
        (-3, 2),
        (0, 5),
        (3, 3),
    ]
    for duration, point in zip(durations, points):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = test_path.header.stamp + rospy.Duration(duration)
        pose_stamped.pose.position.x = point[0]
        pose_stamped.pose.position.y = point[1]
        test_path.poses.append(pose_stamped)
    rospy.wait_for_service("smooth_path")
    smooth_path = rospy.ServiceProxy("smooth_path", SmoothPath)
    smooth_time = -time.perf_counter()
    response = smooth_path(test_path, 0.1)
    smooth_time += time.perf_counter()
    print(f"Test time: {smooth_time:.3f}")
    xs, ys = tuple(zip(*((pose.pose.position.x, pose.pose.position.y) for pose in response.smoothed_path.poses)))
    plt.plot(*tuple(zip(*points)), "o", xs, ys, "x")
    plt.show()
