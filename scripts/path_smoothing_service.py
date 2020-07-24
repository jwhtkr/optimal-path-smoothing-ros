#!/usr/bin/env python
from __future__ import print_function

from path_smoothing.srv import SmoothPath,SmoothPathResponse
from path_smoothing.smooth_traj_opt import smooth_constrained
from tools.multi_array import multi_array_to_array, array_to_multi_array
import rospy
import numpy as np

def parse_request(req):
    desired_path = multi_array_to_array(req.desired_path)
    Q = multi_array_to_array(req.Q)
    R = multi_array_to_array(req.R)
    S = multi_array_to_array(req.S)
    A = multi_array_to_array(req.A)
    b = multi_array_to_array(req.b)
    time_step = req.time_step
    return (desired_path, Q, R, S, A, b, time_step)

def handle_path_smoothing(req):
    (desired_path, Q, R, S, A, b, time_step) = parse_request(req)
    # temp fix for bug
    time_step = 0.01
    smoothed = smooth_constrained(desired_path, Q, R, S, A, b, time_step)

    path = SmoothPathResponse(smoothed_path=smoothed[0],smoothed_path_snaps=smoothed[1])
    return path

def path_smoothing_server():
    rospy.init_node('path_smoothing')
    s = rospy.Service('smooth_path', SmoothPath, handle_path_smoothing)
    rospy.spin()

if __name__ == "__main__":
    path_smoothing_server()
