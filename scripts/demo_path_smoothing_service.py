#!/usr/bin/env python
"""Sends demo data to path smoothing ROS node."""

from __future__ import print_function

from path_smoothing.srv import SmoothPath
# pylint: disable=import-error, no-name-in-module
from tools.multi_array import array_to_multi_array
# pylint: enable=import-error, no-name-in-module
import rospy
import sys
import os.path
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def unpack_input(input_data):
    """
    Unpack input data from a MATLAB file.

    Parameters
    ----------
    input_data : dict
        Input data to unpack.

    Returns
    -------
    (xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, dt) : (numpy.array, ..., float)
        Unpacked data.
    """
    xd_mat = array_to_multi_array(input_data["xd_mat"][:, :-1, :])
    q_mat = array_to_multi_array(input_data["Q"])
    r_mat = array_to_multi_array(input_data["R"])
    s_mat = array_to_multi_array(input_data["S"])
    a_mat = array_to_multi_array(input_data["A"])
    b_mat = array_to_multi_array(input_data["b"])
    dt = input_data["dt"]
    return (xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, dt)

def demo_path_smoothing(input_file):
    """
    Send demo data to path smoothing ROS node.

    Parameters
    ----------
    input_file : string
        Input file to read data from.
    """
    input_path = os.path.abspath(input_file)
    print(input_path)
    input_data = scipy.io.loadmat(input_path)
    (xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, dt) = unpack_input(input_data)

    rospy.wait_for_service('smooth_path')
    try:
        smooth_path = rospy.ServiceProxy('smooth_path', SmoothPath)
        result = smooth_path(xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, dt)
        smoothed_path = np.array(result.smoothed_path)
        # smoothed_path_snaps = np.array(result.smoothed_path_snaps)
        xd_mat = input_data["xd_mat"]
        x_mat = smoothed_path.reshape((xd_mat.shape[0], xd_mat.shape[1]-1,
                                  xd_mat.shape[2]), order="F")
        plt.plot(xd_mat[0, 0, :], xd_mat[1, 0, :],
                 x_mat[0, 0, :], x_mat[1, 0, :])
        plt.show()
    except rospy.ServiceException as ex:
        print("Service call failed: %s"%ex)

def usage():
    """
    Return the usage of this script.

    Returns
    ----------
    string
        Usage info.
    """
    return "%s [input_file]"%sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 2:
        input_file_arg = sys.argv[1]
    else:
        print(usage())
        sys.exit(1)
    demo_path_smoothing(input_file_arg)
