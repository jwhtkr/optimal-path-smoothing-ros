"""Contains functions for matlab to call to smooth a trajectory."""


import os.path

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil

import path_smoothing.smooth_path as smooth
import optimal_control.objectives.quadratic_cost as quad_cost
import optimal_control.constraints.linear_constraints as lin_consts
import optimal_control.solvers.osqp_solver as osqp


def _path_from_mat(path_matrix, time_step):
    def path(t):
        _, n_step = path_matrix.shape
        sub_index = t/time_step
        left = int(max(min(floor(sub_index), n_step), 0))
        right = int(max(min(ceil(sub_index), n_step), 0))
        alpha = (t-left*time_step)/time_step
        x_prev = path_matrix[:, left]
        x_next = path_matrix[:, right]
        return alpha*(x_next - x_prev) + x_prev
    return path


def smooth_unconstrained(desired_path, q_mat, r_mat, s_mat, time_step):
    """Smooth unconstrained."""
    initial_state = desired_path[:, :, 0]
    n_dim, n_int, n_step = desired_path.shape

    path = _path_from_mat(desired_path.reshape(-1, n_step, order="F"),
                          time_step)
    cost = quad_cost.ContinuousQuadraticCost(q_mat, r_mat, s_mat, desired_state=path)

    const = lin_consts.LinearTimeInstantConstraint(time_step*(n_step-1),
                                                   (np.eye(N=n_dim,
                                                           M=n_dim*n_int),
                                                    np.zeros((n_dim, n_dim))),
                                                   desired_path[:, 0, -1, np.newaxis])

    constraints = lin_consts.LinearConstraints(eq_constraints=[const])

    solver = osqp.OSQP()

    smoother = smooth.SmoothPathLinear(constraints, cost, solver, n_step,
                                       initial_state, time_step=time_step)

    return smoother.solve()

def smooth_constrained(desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, time_step):
    """Smooth constrined."""
    initial_state = desired_path[:, :, 0]
    n_dim, n_int, n_step = desired_path.shape

    path = _path_from_mat(desired_path.reshape(-1, n_step, order="F"),
                          time_step)
    cost = quad_cost.ContinuousQuadraticCost(q_mat, r_mat, s_mat, desired_state=path)

    term_const = lin_consts.LinearTimeInstantConstraint(time_step*(n_step-1),
                                                        (np.eye(N=n_dim*n_int,
                                                                M=n_dim*n_int),
                                                         np.zeros((n_dim*n_int, n_dim))),
                                                        desired_path[:, :, -1]
                                                            .reshape(-1, 1, order="F"))
    a_mat = a_mat.reshape(-1, n_dim*(n_int+1), n_step, order="F")
    def ineq_mats(t):
        """Return (A, B) of the constraint at time `t`."""
        ind = int(t//time_step)
        return a_mat[:, :-n_dim, ind], a_mat[:, -n_dim:, ind]
    def ineq_bound(t):
        """Return b of the constraint at time `t`."""
        ind = int(t//time_step)
        return b_mat[:, ind, np.newaxis]
    ineq_const = lin_consts.LinearInequalityConstraint(ineq_mats=ineq_mats,
                                                       bound=ineq_bound)

    constraints = lin_consts.LinearConstraints(eq_constraints=[term_const],
                                               ineq_constraints=[ineq_const])

    smoother = smooth.SmoothPathLinear(constraints, cost, osqp.OSQP(), n_step,
                                       initial_state, time_step=time_step)

    return smoother.solve()

def test_unconstrained():
    """Test unconstrained."""
    f_dir = os.path.dirname(__file__)
    f_name = os.path.join(f_dir, "traj_data.mat")
    tmp = scipy.io.loadmat(f_name)
    xd_mat = tmp["xd_mat"]
    q_mat = tmp["Q"]
    r_mat = tmp["R"]
    s_mat = tmp["S"]
    dt = tmp["dt"][0][0]
    result = smooth_unconstrained(xd_mat[:, :-1, :], q_mat, r_mat, s_mat, dt)
    x_mat = result[0].reshape(xd_mat.shape[0], xd_mat.shape[1]-1,
                              xd_mat.shape[2], order="F")
    plt.plot(xd_mat[0, 0, :], xd_mat[1, 0, :],
             x_mat[0, 0, :], x_mat[1, 0, :])
    plt.show()
    return result


def test_constrained():
    """Test constrained."""
    f_dir = os.path.dirname(__file__)
    f_name = os.path.join(f_dir, "traj_data.mat")
    tmp = scipy.io.loadmat(f_name)
    xd_mat = tmp["xd_mat"]
    q_mat = tmp["Q"]
    r_mat = tmp["R"]
    s_mat = tmp["S"]
    a_mat = tmp["A"]
    b_mat = tmp["b"]
    dt = tmp["dt"][0][0]
    result = smooth_constrained(xd_mat[:, :-1, :], q_mat, r_mat, s_mat, a_mat, b_mat, dt)
    # x_mat = result[0].reshape(xd_mat.shape[0], xd_mat.shape[1]-1,
    #                           xd_mat.shape[2], order="F")
    # plt.plot(xd_mat[0, 0, :], xd_mat[1, 0, :],
    #          x_mat[0, 0, :], x_mat[1, 0, :])
    # plt.show()
    return result


if __name__ == "__main__":
    import time
    t_start = time.time()
    # test_unconstrained()
    test_constrained()
    print(time.time()-t_start)
