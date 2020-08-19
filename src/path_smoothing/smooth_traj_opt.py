"""Contains functions for matlab to call to smooth a trajectory."""

# from __future__ import print_function

import os.path
import collections

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math

import geometry.polytope as poly

import path_smoothing.smooth_path as smooth
import optimal_control.objectives.quadratic_cost as quad_cost
import optimal_control.constraints.linear_constraints as lin_consts
import optimal_control.constraints.binary_constraints as bin_consts
import optimal_control.constraints.constraints as constrs

# pylint: disable=unused-import
import optimal_control.constraints.constraint as constr
import optimal_control.solvers.osqp_solver as osqp
import optimal_control.solvers.gurobi_solver as gurobi
# pylint: enable=unused-import

MAT_FILE_BASE = "traj_data_{}.mat"
MAT_FILES = {"box": MAT_FILE_BASE.format("box"),
             "box_vels": MAT_FILE_BASE.format("box_vels"),
             "voronoi": MAT_FILE_BASE.format("voronoi"),
             "voronoi_vels": MAT_FILE_BASE.format("voronoi_vels"),
             "original": MAT_FILE_BASE.format("original")}
MAT_FILE = MAT_FILES["box"]

SmoothingArguments = collections.namedtuple("SmoothingArguments",
                                            ["constraints",
                                             "cost",
                                             "solver",
                                             "n_step",
                                             "initial_state",
                                             "time_step",
                                             "n_dim",
                                             "n_int"])
SmoothingArgumentsObstacles = collections.namedtuple("SmoothingArgumentsObstacles",
                                                     ["constraints",
                                                      "obstacle_constraints",
                                                      "cost",
                                                      "solver",
                                                      "n_step",
                                                      "initial_state",
                                                      "time_step",
                                                      "n_dim",
                                                      "n_int"])
FreeRegion = collections.namedtuple("FreeRegion", ["A", "b"])

def _path_from_mat(path_matrix, time_step):
    def path(t):
        _, n_step = path_matrix.shape # bug in pylint with nested functions pylint: disable=unused-variable
        sub_index = t/time_step
        left = int(max(min(math.floor(sub_index), n_step), 0))
        right = int(max(min(math.ceil(sub_index), n_step), 0))
        alpha = (t-left*time_step)/time_step
        x_prev = path_matrix[:, left]
        x_next = path_matrix[:, right]
        return alpha*(x_next - x_prev) + x_prev
    return path

def smooth_unconstrained(desired_path, q_mat, r_mat, s_mat, time_step):
    """Smooth unconstrained."""
    args = _setup_unconstrained(desired_path, q_mat, r_mat, s_mat, time_step)

    smoother = smooth.SmoothPathLinear(args.constraints, args.cost,
                                       args.solver, args.n_step,
                                       args.initial_state,
                                       time_step=args.time_step)

    return smoother.solve()

def _setup_unconstrained(desired_path, q_mat, r_mat, s_mat, time_step):
    initial_state = desired_path[:, :, 0]
    n_dim, n_int, n_step = desired_path.shape

    path = _path_from_mat(desired_path.reshape(-1, n_step, order="F"),
                          time_step)
    cost = quad_cost.ContinuousQuadraticCost(q_mat, r_mat, s_mat, desired_state=path)

    term_const = lin_consts.LinearTimeInstantConstraint(time_step*(n_step-1),
                                                        (np.eye(n_dim*n_int),
                                                         np.zeros((n_dim*n_int,
                                                                   n_dim))),
                                                        desired_path[:, :, -1]
                                                            .reshape(-1, 1, order="F"))
    # const = lin_consts.LinearTimeInstantConstraint(time_step*(n_step-1),
    #                                                (np.eye(N=n_dim,
    #                                                        M=n_dim*n_int),
    #                                                 np.zeros((n_dim, n_dim))),
    #                                                desired_path[:, 0, -1, np.newaxis])

    constraints = lin_consts.LinearConstraints(eq_constraints=[term_const])

    # solver = osqp.OSQP()
    solver = gurobi.Gurobi()

    return SmoothingArguments(constraints, cost, solver, n_step, initial_state,
                              time_step, n_dim, n_int)

def smooth_constrained(desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, time_step):
    """Smooth constrined."""
    args = _setup_constrained(desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, time_step)

    smoother = smooth.SmoothPathLinear(args.constraints, args.cost, args.solver,
                                       args.n_step, args.initial_state,
                                       time_step=args.time_step)

    return smoother.solve()

def _setup_constrained(desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, time_step):
    params = _setup_unconstrained(desired_path, q_mat, r_mat, s_mat,
                                         time_step)

    a_mat = a_mat.reshape(-1, params.n_dim*(params.n_int+1), params.n_step, order="F")
    def ineq_mats(t):
        """Return (A, B) of the constraint at time `t`."""
        ind = int(t//time_step)
        return a_mat[:, :-params.n_dim, ind], a_mat[:, -params.n_dim:, ind]
    def ineq_bound(t):
        """Return b of the constraint at time `t`."""
        ind = int(t//time_step)
        return b_mat[:, ind, np.newaxis]
    ineq_const = lin_consts.LinearInequalityConstraint(ineq_mats=ineq_mats,
                                                       bound=ineq_bound)

    constraints = lin_consts.LinearConstraints(ineq_constraints=[ineq_const],
                                               constraints=[params.constraints])

    return SmoothingArguments(constraints, params.cost, params.solver,
                              params.n_step, params.initial_state,
                              params.time_step, params.n_dim, params.n_int)

def smooth_obstacles_mip(desired_path, q_mat, r_mat, s_mat, a_cnstr_mat,
                         b_cnstr_mat, free_regions, time_step):
    """Smooth the path in the presence of obstacles."""
    args = _setup_obstacles_mip(desired_path, q_mat, r_mat, s_mat, a_cnstr_mat,
                                b_cnstr_mat, free_regions, time_step)

    smoother = smooth.SmoothPathLinearObstacles(args.constraints,
                                                args.obstacle_constraints,
                                                args.cost, args.solver,
                                                args.n_step, args.initial_state,
                                                time_step=args.time_step)

    return smoother.solve()

def _setup_obstacles_mip(desired_path, q_mat, r_mat, s_mat, a_cnstr_mat,
                         b_cnstr_mat, free_regions, time_step):
    params = _setup_constrained(desired_path, q_mat, r_mat, s_mat, a_cnstr_mat,
                                b_cnstr_mat, time_step)

    solver = gurobi.Gurobi()  # Only Gurobi can handle obstacles (Mixed Integer Programming)

    constr_list = []
    for i, region in enumerate(free_regions):
        a_pos_mat, b_vec = region
        a_rest_mat = np.zeros((a_pos_mat.shape[0],
                               params.n_dim*(params.n_int-1)))
        a_mat = np.concatenate([a_pos_mat, a_rest_mat], axis=1)
        ineq_mats = lambda t, a=a_mat: (a, np.zeros((a.shape[0], params.n_dim)))
        inequality = lin_consts.LinearInequalityConstraint(ineq_mats,
                                                           lambda t, b=b_vec: b)
        big_m_vec = _calc_big_m(region, np.array([[-100, 100], [-100, 100]]))
        bin_const = bin_consts.ImplicationInequalityConstraint(i,
                                                               inequality,
                                                               big_m_vec,
                                                               len(free_regions))
        constr_list.append(bin_const)

    return SmoothingArgumentsObstacles(params.constraints, constr_list,
                                       params.cost, solver, params.n_step,
                                       params.initial_state, params.time_step,
                                       params.n_dim, params.n_int)

def _calc_big_m(region, limits):
    a_mat, b_vec = region
    expanded_limits = _expand_limits(limits)  # expand limits to include all
                                              # combos of the limit values.
                                              # This is combinatorial,so careful!
    values = a_mat @ expanded_limits
    return np.max(np.abs(values), axis=1) + np.abs(b_vec)

def _expand_limits(limits):
    """Expand `limits` to include all the "corners" of a hyper-cube."""
    if limits.shape[0] == 1:
        return limits
    else:
        prev_limits = _expand_limits(limits[:-1, :])
        lower_limit = limits[-1, 0] * np.ones((prev_limits.shape[1]))
        upper_limit = limits[-1, 1] * np.ones((prev_limits.shape[1]))
        return np.block([[prev_limits, prev_limits], [lower_limit, upper_limit]])


def test_unconstrained():
    """Test unconstrained."""
    xd_mat, q_mat, r_mat, s_mat, dt = _unconstrained_from_mat(_load())
    result = smooth_unconstrained(xd_mat[:, :-1, :], q_mat, r_mat, s_mat, dt)
    _plot(result[0], xd_mat)
    return result

def _load():
    f_dir = os.path.dirname(__file__)
    f_name = os.path.join(f_dir, MAT_FILE)
    return scipy.io.loadmat(f_name)

def _plot(x, xd_mat):
    x_mat = x.reshape(xd_mat.shape[0], xd_mat.shape[1]-1,
                      xd_mat.shape[2], order="F")
    plt.plot(xd_mat[0, 0, :], xd_mat[1, 0, :],
             x_mat[0, 0, :], x_mat[1, 0, :])
    plt.show()

def _unconstrained_from_mat(args):
    return args["xd_mat"], args["Q"], args["R"], args["S"], args["dt"][0][0]

def test_constrained():
    """Test constrained."""
    xd_mat, q_mat, r_mat, s_mat, a_mat, b_vec, dt = _constrained_from_mat(_load())
    result = smooth_constrained(xd_mat[:, :-1, :], q_mat, r_mat, s_mat, a_mat,
                                b_vec, dt)
    _plot(result[0], xd_mat)
    return result

def _constrained_from_mat(args):
    xd_mat, q_mat, r_mat, s_mat, dt = _unconstrained_from_mat(args)
    return xd_mat, q_mat, r_mat, s_mat, args["A"], args["b"], dt

def test_obstacles_mip():
    """Test MIP Obstacle avoidance."""
    xd_mat, q_mat, r_mat, s_mat, a_mat, b_vec, dt = _constrained_from_mat(_load())
    free_regions = _create_free_regions(np.squeeze(xd_mat[:, 0, :]))
    # fig, ax = _plot_free_regions(free_regions)
    # ax.plot(xd_mat[0, 0, :].flatten(), xd_mat[1, 0, :].flatten())
    # plt.show()

    result = smooth_obstacles_mip(xd_mat[:, :-1, :], q_mat, r_mat, s_mat, a_mat,
                                  b_vec, free_regions, dt)
    # _plot(result[0], xd_mat)
    return result

def _create_free_regions(position, dist=5.0):
    offset = np.array([[dist, -dist, -dist, dist], [dist, dist, -dist, -dist]])
    curr_pos = position[:, 0].reshape(-1, 1)
    curr_idx = 0
    polys = []
    while curr_idx < position.shape[1]:
        box_vertices = np.transpose(curr_pos+offset)
        polys.append(poly.Polytope.from_vertices_rays(box_vertices))
        tmp_idx, tmp_pos = _get_next_after(position, curr_pos, curr_idx,
                                           dist - dist/20)
        curr_idx, curr_pos = _get_next_after(position, tmp_pos, tmp_idx,
                                             dist - dist/20)


    # _plot_free_regions(polys)
    # plt.show()
    return [FreeRegion(*region.get_a_b()) for region in polys]

def _get_next_after(position, curr_pos, curr_idx, dist):
    dist_vec = np.linalg.norm(position-curr_pos, axis=0)
    far_idx = np.nonzero(dist_vec > dist - dist/10)[0]
    after_idx = np.nonzero(far_idx > curr_idx)[0]
    try:
        out_idx = far_idx[after_idx[0]]
    except IndexError:  # there are no indices in `far_idx` greater than the current index
        if curr_idx < position.shape[1] - 1:  # If currently less than last index
            out_idx = position.shape[1] - 1  # Set to last index
        else:  # currently at or beyond last valid index
            out_idx = position.shape[1]  # Set to an invalid index value
    try:
        out_pos = position[:, out_idx].reshape(-1, 1)
    except IndexError:
        out_pos = curr_pos # if the index is out of bounds, then return current position

    return out_idx, out_pos

def _plot_free_regions(regions):
    polys = []
    for region in regions:
        if not isinstance(region, poly.Polytope):
            region = poly.Polytope.from_a_b(region.A, region.b)
        polys.append(plt.Polygon(region.get_vertices_rays()[0]))
    poly_collection = plt.matplotlib.collections.PatchCollection(polys,
                                                                 alpha=0.4)
    fig, ax = plt.subplots()
    ax.add_collection(poly_collection)
    ax.autoscale()
    return fig, ax

if __name__ == "__main__":
    import time
    t_start = time.time()
    # test_unconstrained()
    # test_constrained()
    test_obstacles_mip()
    print(time.time()-t_start)
