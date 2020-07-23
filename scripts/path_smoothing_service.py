#!/usr/bin/env python
from __future__ import print_function

from path_smoothing.srv import SmoothPath,SmoothPathResponse
from std_msgs.msg import Float32MultiArray
from path_smoothing.smooth_traj_opt import smooth_constrained
import rospy
import numpy as np

def handle_path_smoothing(req):
    desired_path = req.desired_path
    print(desired_path)

    offset = desired_path.layout.data_offset

    dim = desired_path.layout.dim
    dim = map(lambda x: x.size, dim)
    dim = tuple(dim)
    print(dim)

    desired_path = np.array(desired_path.data[offset:])
    print(desired_path)
    desired_path = desired_path.reshape(dim)
    print(desired_path)
    print(desired_path.shape)
    # in:     
    # desired_path
    # Q
    # R
    # S
    # A_mat
    # b_mat
    # time_step
    #
    # out:
    # smoothed_path

    array = Float32MultiArray(data=[0, 1, 2, 3, 4, 5])
    path = SmoothPathResponse(smoothed_path=array)
    return path

def path_smoothing_server():
    rospy.init_node('path_smoothing')
    s = rospy.Service('smooth_path', SmoothPath, handle_path_smoothing)
    rospy.spin()

if __name__ == "__main__":
    path_smoothing_server()
