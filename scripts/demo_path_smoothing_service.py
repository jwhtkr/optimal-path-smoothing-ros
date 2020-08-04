#!/usr/bin/env python
from __future__ import print_function

from path_smoothing.srv import SmoothPath,SmoothPathResponse
from path_smoothing.smooth_traj_opt import smooth_constrained
from tools.multi_array import array_to_multi_array
import rospy
import sys
import os.path
import scipy
import numpy as np
import matplotlib.pyplot as plt

def unpack_input(input_data):
    xd_mat = array_to_multi_array(input_data["xd_mat"][:, :-1, :])
    Q = array_to_multi_array(input_data["Q"])
    R = array_to_multi_array(input_data["R"])
    S = array_to_multi_array(input_data["S"])
    A = array_to_multi_array(input_data["A"])
    b = array_to_multi_array(input_data["b"])
    dt = input_data["dt"]
    return (xd_mat, Q, R, S, A, b, dt)

def demo_path_smoothing(input_file):
    input_path = os.path.abspath(input_file)
    print(input_path)
    input_data = scipy.io.loadmat(input_path)
    (xd_mat, Q, R, S, A, b, dt) = unpack_input(input_data)

    rospy.wait_for_service('smooth_path')
    try:
        smooth_path = rospy.ServiceProxy('smooth_path', SmoothPath)
        result = smooth_path(xd_mat, Q, R, S, A, b, dt)
        smoothed_path = np.array(result.smoothed_path)
        smoothed_path_snaps = np.array(result.smoothed_path_snaps)
        xd_mat = input_data["xd_mat"]
        x_mat = smoothed_path.reshape(xd_mat.shape[0], xd_mat.shape[1]-1,
                                  xd_mat.shape[2], order="F")
        plt.plot(xd_mat[0, 0, :], xd_mat[1, 0, :],
                 x_mat[0, 0, :], x_mat[1, 0, :])
        plt.show()
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def usage():
    return "%s [input_file]"%sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
    else:
        print(usage())
        sys.exit(1)
    demo_path_smoothing(input_file)
