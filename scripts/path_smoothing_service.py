#!/usr/bin/env python
"""ROS node providing path smoothing service."""

from __future__ import print_function
import array

import gurobipy as gp
import numpy as np
import rospy

from path_smoothing.srv import SmoothPath, SmoothPathResponse, SmoothTraj
from tools.ros_paths import path_to_traj, traj_to_path
from tools.multi_array import array_to_multi_array

# def parse_request(req):
#     """
#     Parse a given ROS request.

#     Parameters
#     ----------
#     req : SmoothTraj
#         Request to parse.

#     Returns
#     -------
#     (desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, free_regions, time_step) : (numpy.ndarray, ..., list of tuples of numpy.ndarray, float)
#         Parsed request data.
#     """
#     desired_path = multi_array_to_array(req.desired_path)
#     q_mat = multi_array_to_array(req.Q)
#     r_mat = multi_array_to_array(req.R)
#     s_mat = multi_array_to_array(req.S)
#     a_mat = multi_array_to_array(req.A)
#     b_mat = multi_array_to_array(req.b)
#     regions_A = [multi_array_to_array(A) for A in req.regions_A]
#     regions_b = [multi_array_to_array(b) for b in req.regions_b]
#     free_regions = list(zip(regions_A, regions_b))
#     time_step = req.time_step
#     return (desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, free_regions, time_step)


def handle_path_smoothing(req: SmoothPath._request_class, traj_smoother):
    """
    Handle a ROS path smoothing request.

    Parameters
    ----------
    req : SmoothPath
        Request to handle.

    Returns
    -------
    path : SmoothPathResponse
        Smoothed path.
    """
    desired_traj = path_to_traj(req.desired_path, 2, req.time_step)
    n_dim, n_int, n_step = desired_traj.shape
    q_mat = array_to_multi_array(np.eye(n_dim * n_int))
    r_mat = array_to_multi_array(np.eye(n_dim))
    s_mat = q_mat
    a_mat = array_to_multi_array(np.empty((0, n_dim, n_int + 1, n_step)))
    b_mat = array_to_multi_array(np.empty((0,)))
    try:
        smoothed = traj_smoother(
            array_to_multi_array(desired_traj),
            q_mat,
            r_mat,
            s_mat,
            a_mat,
            b_mat,
            [],
            [],
            req.time_step,
        )
    except gp.GurobiError as ex:
        raise rospy.ServiceException(ex)
    return SmoothPathResponse(
        smoothed_path=traj_to_path(
            np.reshape(smoothed.smoothed_traj, desired_traj.shape, order="F"),
            req.time_step,
            req.desired_path.header.stamp,
        )
    )


def path_smoothing_server():
    """ROS node providing smooth_path service."""
    rospy.init_node("path_smoothing_server")
    rospy.wait_for_service("smooth_traj")
    traj_smoother = rospy.ServiceProxy("smooth_traj", SmoothTraj)
    rospy.Service(
        "smooth_path", SmoothPath, lambda req: handle_path_smoothing(req, traj_smoother)
    )
    rospy.spin()


if __name__ == "__main__":
    path_smoothing_server()
