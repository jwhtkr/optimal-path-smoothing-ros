"""Utilities to deal with ROS Path messages."""

import math

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav_msgs.msg import Path
import numpy as np
import rospy


def path_to_traj(path: Path, n_int: int, time_step: float):
    """Convert a path message to a trajectory. Assumes 2D."""
    if path.poses is None:
        raise ValueError("An empty path was given.")
    if n_int < 1:
        raise ValueError("A non-positive value was given for n_int")

    time_start = path.header.stamp.to_sec()
    time_end = path.poses[-1].header.stamp.to_sec()
    horizon = time_end - time_start
    n_step = math.floor(horizon / time_step)
    times = [i * time_step for i in range(n_step)]

    path_points = []
    path_times = []
    for pose_stamped in path.poses:
        path_times.append(pose_stamped.header.stamp.to_sec())
        position = pose_stamped.pose.position
        path_points.append([position.x, position.y])

    traj_points = interp(times, path_times, path_points)
    traj = []
    for idx, (_, point) in enumerate(zip(times, traj_points)):
        traj_val = []
        for j in range(n_int):
            if j == 0:
                # add position
                traj_val.append(point)
                continue
            if j == 1 and idx < len(traj_points) - 1:
                # add velocity
                traj_val.append(calc_velocity(point, traj_points[idx + 1], time_step))
                continue
            traj_val.append([0, 0])
        traj.append(traj_val)

    traj_arr = np.array(traj)
    return traj_arr.T


def traj_to_path(traj, time_step: float, stamp) -> Path:
    path = Path()
    path.header.stamp = stamp
    path.poses = []

    n_dim, n_int, n_step = traj.shape
    for i in range(n_step):
        time_stamp = stamp + rospy.Duration.from_sec(time_step*i)
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = time_stamp
        pose_stamped.pose.position = Point(x=traj[0,0,i], y=traj[1,0,i], z=0)
        pose_stamped.pose.orientation = calc_quaternion(traj[:,:,i])
        path.poses.append(pose_stamped)

    return path


def calc_quaternion(traj_point) -> Quaternion:
    theta = math.atan2(traj_point[1,1].item(), traj_point[0,1].item())
    return Quaternion(x=0, y=0, z=math.sin(theta/2), w= math.cos(theta/2))


def interp(desired_times, known_times, known_points):
    points = np.array(known_points)

    desired_points = []
    known_idx = 0
    for time in desired_times:
        while time > known_times[known_idx + 1]:
            known_idx += 1
        time_percentage = (time - known_times[known_idx]) / (
            known_times[known_idx + 1] - known_times[known_idx]
        )
        desired_points.append(
            (
                (1 - time_percentage) * points[known_idx]
                + time_percentage * points[known_idx + 1]
            ).tolist()
        )
    return desired_points


def calc_velocity(point, next_point, time_step):
    point = np.array(point)
    next_point = np.array(next_point)
    return ((next_point - point) / time_step).tolist()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

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

    trajectory = path_to_traj(test_path, 2, 0.1)
    xs, ys = tuple(zip(*points))
    plt.plot(xs, ys, "o", trajectory[0, 0, :], trajectory[1, 0, :], "x")
    plt.show()
