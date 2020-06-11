"""Contains functions for matlab to call to smooth a trajectory."""


import os.path

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

import path_smoothing.smooth_path as smooth
import optimal_control.objectives.quadratic_cost as quad_cost
import optimal_control.constraints.linear_constraints as lin_consts
import optimal_control.solvers.osqp_solver as osqp


def _path_from_mat(path_matrix, time_step):
    def path(t):
        ind_prev = int(t//time_step)
        alpha = (t-ind_prev*time_step)/time_step
        x_prev = path_matrix[:, ind_prev]
        x_next = path_matrix[:, ind_prev+1]
        return alpha*(x_next - x_prev) + x_prev
    return path


def smooth_unconstrained(desired_path, Q, R, S, time_step):
    initial_state = desired_path[:, :, 0]
    n_dim, n_int, n_step = desired_path.shape

    path = _path_from_mat(desired_path.reshape(-1, n_step, order="F"),
                          time_step)
    cost = quad_cost.ContinuousQuadraticCost(Q, R, S, desired_state=path)

    constraints = None

    solver = osqp.OSQP()

    smoother = smooth.SmoothPathLinear(constraints, cost, solver, n_step,
                                       initial_state, time_step=time_step)

    return smoother.solve()


def test():
    f_dir = os.path.dirname(__file__)
    f_name = os.path.join(f_dir, "traj_data.mat")
    tmp = scipy.io.loadmat(f_name)
    xd_mat = tmp["xd_mat"]
    Q = tmp["Q"]
    R = tmp["R"]
    S = tmp["S"]
    dt = tmp["dt"][0][0]
    result = smooth_unconstrained(xd_mat[:, :-1, :], Q, R, S, dt)
    x_mat = result[0].reshape(xd_mat.shape[0], xd_mat.shape[1]-1,
                                xd_mat.shape[2], order="F")
    plt.plot(xd_mat[0, 0, :], xd_mat[1, 0, :],
             x_mat[0, 0, :], x_mat[1, 0, :])
    plt.show()
    return result


if __name__ == "__main__":
    # import time
    # t_start = time.time()
    test()
    # print(time.time()-t_start)
