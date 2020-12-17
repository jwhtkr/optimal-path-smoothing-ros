#!/usr/bin/env python
"""ROS node providing path smoothing service."""

from __future__ import print_function

from path_smoothing.srv import SmoothPath, SmoothPathResponse
# pylint: disable=import-error, no-name-in-module
from path_smoothing.smooth_traj_opt import smooth_constrained, _free_regions_from, CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE
import path_smoothing.smooth_path_lp as smooth_path
from tools.multi_array import multi_array_to_array
# pylint: enable=import-error, no-name-in-module
import rospy

def parse_request(req):
    """
    Parse a given ROS request.

    Parameters
    ----------
    req : SmoothPath
        Request to parse.

    Returns
    -------
    (desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, free_regions, time_step) : (numpy.ndarray, ..., list of tuples of numpy.ndarray, float)
        Parsed request data.
    """
    desired_path = multi_array_to_array(req.desired_path)
    q_mat = multi_array_to_array(req.Q)
    r_mat = multi_array_to_array(req.R)
    s_mat = multi_array_to_array(req.S)
    a_mat = multi_array_to_array(req.A)
    b_mat = multi_array_to_array(req.b)
    regions_A = [multi_array_to_array(A) for A in req.regions_A]
    regions_b = [multi_array_to_array(b) for b in req.regions_b]
    free_regions = list(zip(regions_A, regions_b))
    time_step = req.time_step
    return (desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, free_regions, time_step)

def handle_path_smoothing(req):
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
    # smooth = smooth_constrained
    # smooth = lambda *args: smooth_path.smooth_path_qp(*args[:-1], _free_regions_from(CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE), args[-1])
    # smooth = lambda *args: smooth_path.smooth_path_lp(*args[:-1], _free_regions_from(CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE), args[-1])
    smooth = smooth_path.smooth_path_qp


    (desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, free_regions, time_step) = parse_request(req)
    try:
        smoothed = smooth(desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, free_regions, time_step)
        path = SmoothPathResponse(smoothed_path=smoothed[0], smoothed_path_snaps=smoothed[1])
    except smooth_path.gp.GurobiError as ex:
        raise rospy.ServiceException(ex)
    return path

def path_smoothing_server():
    """ROS node providing smooth_path service."""
    rospy.init_node('path_smoothing')
    rospy.Service('smooth_path', SmoothPath, handle_path_smoothing)
    rospy.spin()

if __name__ == "__main__":
    path_smoothing_server()
