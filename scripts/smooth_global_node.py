#! /usr/bin/env python
"""Smooth a global plan."""
import math
import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
# from tf.transformations import quaternion_from_euler
from path_smoothing.srv import SmoothPath, GetSmoothPlan
from rrt_server.srv import GetPlan

# pylint: disable=import-error, no-name-in-module
from tools.multi_array import array_to_multi_array, multi_array_to_array
from path_smoothing.smooth_path_lp import smooth_path_qp as smooth
# pylint: disable=import-error, no-name-in-module


def get_original_path(start_pose, end_pose):
    get_plan_topic = rospy.get_param("get_plan_topic", default="rrt_server")
    rospy.wait_for_service(get_plan_topic)
    try:
        get_plan = rospy.ServiceProxy(get_plan_topic, GetPlan)
        plan = get_plan(planner_type=GetPlan._request_class.RRT,
                        starting_pose=start_pose, target_pose=end_pose,
                        target_cost=0, max_time=rospy.Duration(2), max_iterations=0)
    except rospy.ServiceException as ex:
        rospy.logwarn("Service call failed: {}".format(ex))

    return plan.solution

def get_free_regions(plan):
    return [], []

def get_smooth_path(start_pose, end_pose):
    original_path = get_original_path(start_pose, end_pose)
    xd_mat = path_to_traj(original_path)
    ndim, nint, nstep = xd_mat.shape
    xd_mat = array_to_multi_array(xd_mat)
    q_mat, r_mat, s_mat = _create_qrs(ndim, nint)
    a_mat = array_to_multi_array(np.empty((0, ndim, nint+1, nstep)))
    b_mat = array_to_multi_array(np.empty((0, nstep)))
    free_regions = get_free_regions(original_path)
    dt = 0.1
    rospy.wait_for_service("smooth_path")
    try:
        smooth_path = rospy.ServiceProxy("smooth_path", SmoothPath)
        result = smooth_path(xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat,
                             free_regions[0], free_regions[1], dt)
    except rospy.ServiceException as ex:
        print("Service call failed: {}".format(ex))
        raise
    return (np.array(result.smoothed_path).reshape((ndim, nint, nstep), order="F"),
            np.array(result.smoothed_path_snaps).reshape((ndim, nstep-1), order="F"))

def _create_qrs(ndim, nint):
    int_weights = [1, 0 , 10, 10, 10]
    q_mat = np.diag([int_weights[j] for _ in range(ndim) for j in range(nint)])
    r_mat = np.diag([int_weights[-1] for _ in range(ndim)])
    s_mat = q_mat
    return array_to_multi_array(q_mat), array_to_multi_array(r_mat), array_to_multi_array(s_mat)

def get_smooth_plan_cb(req):
    start, end = req.starting_pose, req.target_pose
    traj, inputs = get_smooth_path(start, end)
    return traj_to_path(traj), array_to_multi_array(traj), array_to_multi_array(inputs)

def traj_to_path(traj):
    seq = 1
    path = Path()
    path.header.frame_id = "map"
    path.header.stamp = rospy.Time.now()
    path.header.seq = 0
    for k in range(traj.shape[2]):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = path.header.frame_id
        pose_stamped.header.stamp = path.header.stamp
        pose_stamped.header.seq = seq
        pose_stamped.pose.orientation.x = 0
        pose_stamped.pose.orientation.y = 0
        pose_stamped.pose.orientation.z = 0
        pose_stamped.pose.orientation.w = 1
        pose_stamped.pose.position.x = traj[0, 0, k]
        pose_stamped.pose.position.y = traj[1, 0, k]
        if traj.shape[0] == 3:
            pose_stamped.pose.position.z = traj[2, 0, k]
        path.poses.append(pose_stamped)
    return path

def path_to_traj(path, vel=1, nint=4, dt=0.1):
    def dist(a, b):
        return np.linalg.norm(a-b)

    # return path
    nposes = len(path.poses)
    point_arr = np.zeros((3, nposes))

    for i, pose_stamped in enumerate(path.poses):
        point_arr[:,i] = [pose_stamped.pose.position.x,
                          pose_stamped.pose.position.y,
                          pose_stamped.pose.position.z]

    total_dist = 0
    for i in range(nposes-1):
        total_dist += dist(point_arr[:, i], point_arr[:, i+1])
    nstep = int(math.ceil(total_dist/vel/dt))

    traj = np.zeros((3, nint, nstep))
    i_points = 0
    prev_point = point_arr[:, i_points]
    next_point = point_arr[:, i_points + 1]
    delta_dist = dist(prev_point, next_point)
    unit_dir = (next_point - prev_point)/delta_dist
    prev_dist = 0
    for k in range(nstep-1):
        if k*vel*dt >= prev_dist + delta_dist:
            prev_dist += delta_dist
            i_points += 1
            prev_point = point_arr[:, i_points]
            next_point = point_arr[:, i_points + 1]
            delta_dist = dist(prev_point, next_point)
            unit_dir = (next_point - prev_point)/delta_dist

        alpha = (k*vel*dt - prev_dist)/delta_dist
        point = alpha*delta_dist*unit_dir + prev_point
        traj[:, 0, k] = point
        if k != 0:
            traj[:, 1, k] = vel*unit_dir

    traj[:, 0, -1] = point_arr[:, -1]

    if np.allclose(traj[-1, 0, :], 0):
        traj = traj[:-1, :, :]

    return traj


def _start_end(start_pt, end_pt):
    start_ps = Pose()
    start_ps.position.x, start_ps.position.y, start_ps.position.z = start_pt[0], start_pt[1], 0
    # q = quaternion_from_euler(0, 0, 0)
    start_ps.orientation.x, start_ps.orientation.y, start_ps.orientation.z, start_ps.orientation.w = (0., 0., 0., 1.)

    end_ps = Pose()
    end_ps.position.x, end_ps.position.y, end_ps.position.z = end_pt[0], end_pt[1], 0
    end_ps.orientation = start_ps.orientation
    return start_ps, end_ps

if __name__ == "__main__":
    TEST = False
    rospy.init_node("smooth_global")
    rospy.Service("get_global_smooth", GetSmoothPlan, get_smooth_plan_cb)
    if TEST:
        import time
        start = time.time()
        traj, inputs = get_smooth_path(*_start_end((2.5, -22.5), (2.5, 22.5)))
        plan = traj_to_path(traj[:, :, ::10])
        print("Time: {}".format(time.time() - start))
        pub = rospy.Publisher("rrt_plan", Path, latch=True, queue_size=10)
        pub.publish(plan)
        rospy.sleep(0.5)
    else:
        rospy.spin()
