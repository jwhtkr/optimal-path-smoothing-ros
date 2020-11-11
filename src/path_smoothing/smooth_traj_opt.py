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
MAT_FILE = MAT_FILES["voronoi"]

USE_KEY_FRAME = False
KEY_FRAME_STEP = 20

CREATE_FREE_REGIONS = False

Point = collections.namedtuple("Point", ["x", "y"])

OBSTACLE_MIDDLE = {"bl": Point(12.5, 7.),
            "tl": Point(12.5, 8.),
            "tr": Point(13.5, 8.),
            "br": Point(13.5, 7.)}

CORRIDOR_WORLD_STRAIGHT = {1: ((-2., -2., 14., 14.),
                               (-1.75, 1.75, 1.75, -1.75)),
                           2: ((10., 10., 14., 14.,),
                               (-1.75, 10., 10., -1.75)),
                           3: ((10., 10., 22., 22.),
                               (6., 10., 10., 6.)),
                           4: ((18., 18., 22., 22.),
                               (-8., 10., 10., -8)),
                           5: ((0., 0., 22., 22.),
                               (-8., -4., -4., -8.))}

CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE = {1: ((-2., -2., 14., 14.),
                                             (-1.75, 1.75, 1.75, -1.75)),
                                         2: ((10., 10., 14., 14.,),
                                             (-1.75, OBSTACLE_MIDDLE["bl"].y, OBSTACLE_MIDDLE["bl"].y, -1.75)),
                                         3: ((10., 10., OBSTACLE_MIDDLE["tl"].x, OBSTACLE_MIDDLE["tl"].x),
                                             (OBSTACLE_MIDDLE["bl"].y-1, 10, 10, OBSTACLE_MIDDLE["bl"].y-1)),
                                         4: ((10., 10., OBSTACLE_MIDDLE["tr"].x+1, OBSTACLE_MIDDLE["tr"].x+1),
                                             (OBSTACLE_MIDDLE["tl"].y, 10., 10., OBSTACLE_MIDDLE["tl"].y)),
                                         5: ((OBSTACLE_MIDDLE["tr"].x, OBSTACLE_MIDDLE["tr"].x, 22., 22.),
                                             (6., 10., 10., 6.)),
                                         6: ((18., 18., 22., 22.),
                                             (-8., 10., 10., -8)),
                                         7: ((0., 0., 22., 22.),
                                             (-8., -4., -4., -8.))}

WORLD = CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE
# WORLD_SHORT = {1: ((-2., -2., 2., 6.),
#                    (-1.75, 1.75, 1.75, -1.75)),
#                2: ((3., 7., 14., 14.),
#                    (-1.75, 1.75, 1.75, -1.75))}
# WORLD_SHORT = {1: CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE[2],
#                2: CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE[5]}
WORLD_SHORT = {1: CORRIDOR_WORLD_STRAIGHT[2],
               2: CORRIDOR_WORLD_STRAIGHT[3]}

WORLD_SHORT_N = {1: {1: ((10., 10., 16., 22., 22.),
                         (-1.75, 3., 10., 10., -1.75))},
                 2: {1: CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE[2],
                     2: CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE[5]},
                 3: {1: ((10., 10., 14., 14.),
                         (-1.75, 6., 6., -1.75)),
                     2: ((13., 13., 14., 14.),
                         (6., 7., 7., 6.)),
                     3: ((14., 14., 22., 22.,),
                         (6., 10., 10., 6.))}}

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

def _indicator(t, dt):
    return True if ((math.floor(t/dt) % KEY_FRAME_STEP) == 0) else False

def _key_frame(key_frame_return_func, other_return_func, indicator_func):
    def _key_frame_func(*args, **kwargs):
        if indicator_func(*args, **kwargs):
            return key_frame_return_func(*args, **kwargs)
        else:
            return other_return_func(*args, **kwargs)

    return _key_frame_func

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
        ind = math.floor(t/time_step)
        return a_mat[:, :-params.n_dim, ind], a_mat[:, -params.n_dim:, ind]
    def ineq_bound(t):
        """Return b of the constraint at time `t`."""
        ind = math.floor(t/time_step)
        return b_mat[:, ind, np.newaxis]

    if USE_KEY_FRAME:
        ineq_mats = _key_frame(ineq_mats,
                               lambda t: (np.empty((0, params.n_dim*params.n_int)),
                                          np.empty((0, params.n_dim))),
                               lambda t: _indicator(t, params.time_step))
        ineq_bound = _key_frame(ineq_bound,
                                lambda t: np.empty((0, 1)),
                                lambda t: _indicator(t, params.time_step))
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

    smoother = smooth.SmoothPathLinearObstaclesMip(args.constraints,
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

    if USE_KEY_FRAME:
        indicator = lambda t: _indicator(t, params.time_step)
        ineq_empty = lambda t: (np.empty((0, params.n_dim*params.n_int)),
                                np.empty((0, params.n_dim)))
        bound_empty = lambda t: np.empty((0, 1))
        m_vec_empty = lambda t: np.empty((0, 1))

    constr_list = []
    for i, region in enumerate(free_regions):
        a_pos_mat, b_vec = region
        a_rest_mat = np.zeros((a_pos_mat.shape[0],
                               params.n_dim*(params.n_int-1)))
        a_mat = np.concatenate([a_pos_mat, a_rest_mat], axis=1)
        ineq_mats = lambda t, *, _a=a_mat: (_a, np.zeros((_a.shape[0], params.n_dim)))
        bound = lambda t, *, _b=b_vec: _b.reshape(-1, 1)
        big_m_vec = _calc_big_m(region, np.array([[-50, 50], [-50, 50]]))
        big_m_vec = lambda t, *, _m=big_m_vec: _m
        if USE_KEY_FRAME:
            ineq_mats = _key_frame(ineq_mats, ineq_empty, indicator)
            bound = _key_frame(bound, bound_empty, indicator)
            big_m_vec = _key_frame(big_m_vec, m_vec_empty, indicator)
        inequality = lin_consts.LinearInequalityConstraint(ineq_mats, bound)
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
    return (np.max(np.abs(values), axis=1, keepdims=1)
            + np.abs(b_vec.reshape(-1, 1)))

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
    _plot_traj_and_states(result[0], result[1], xd_mat)
    return result

def _load():
    f_dir = os.path.dirname(__file__)
    f_name = os.path.join(f_dir, MAT_FILE)
    return scipy.io.loadmat(f_name)

def _plot_traj_and_states(x, u, xd_mat):
    x_mat = x.reshape(xd_mat.shape[0], xd_mat.shape[1]-1, xd_mat.shape[2],
                      order="F")
    u_mat = u.reshape(xd_mat.shape[0], xd_mat.shape[2]-1, order="F")
    fig = plt.figure()
    fig.set_tight_layout(True)
    fig, axes_traj = _plot_trajectory(x_mat, xd_mat, fig, fig.add_subplot(121))
    fig, axes_states = _plot_states(x_mat, u_mat, xd_mat, fig)
    return fig, axes_traj

def _plot_trajectory(x, xd_mat, fig=None, axes=None):
    if not fig and not axes:
        fig, axes = plt.subplots()
    if not axes:
        axes = fig.add_subplot(111)
    axes.plot(xd_mat[0, 0, :], xd_mat[1, 0, :], x[0, 0, :], x[1, 0, :])
    axes.axis("equal")
    return fig, axes

def _plot_states(x, u, xd_mat, fig=None):
    if not fig:
        fig = plt.figure()
    ndim, nint, nstep = x.shape
    ncol = 2*ndim
    nrow = nint + 1

    for j in range(ndim):
        for i in range(nint):
            row = i
            col = ndim + j
            axes = fig.add_subplot(nrow, ncol, row*ncol+col+1, xticklabels=[])
            if i == 0:
                axes.set_title(["x", "y"][j])
            if j == 0 and i > 0:
                axes.set_ylabel([r"$\frac{d}{dt}$", r"$\frac{d^2}{dt^2}$", r"$\frac{d^3}{dt^3}$"][i-1], rotation="horizontal")
            axes.plot(xd_mat[j, i, :])
            axes.plot(x[j, i, :])
    for j in range(ndim):
        row = nrow - 1
        col = ndim + j
        axes = fig.add_subplot(nrow, ncol, row*ncol+col+1)
        if j == 0:
            axes.set_ylabel(r"$\frac{d^4}{dt^4}$", rotation="horizontal")
        axes.set_xlabel("time step index")
        axes.plot(xd_mat[j, -1, :])
        axes.plot(u[j, :])

    return fig, axes

def _plot_characteristics(x, xd_mat, fig=None):
    if not fig:
        fig = plt.figure()
    ndim, nint, nstep = x.shape
    ncol = 2*ndim + 2
    nrow = 3
    funcs = [[calc_curvature], [calc_vel, calc_omega], [calc_accel, calc_alpha]]
    ylabels = [["$\\kappa$"], ["v", "$\\omega$"], ["a", "$\\alpha$"]]
    for i in range(3):
        for j in range(2):
            row = i
            col = 2*ndim + j
            if i == 0:
                if j == 1:
                    continue
                axes = fig.add_subplot(nrow, 3, row*3+3)
            else:
                axes = fig.add_subplot(nrow, ncol, row*ncol+col+1)

            axes.set_ylabel(ylabels[i][j])
            if i == 2:
                axes.set_xlabel("time step index")

            axes.plot(funcs[i][j](xd_mat[:, :-1, :]))
            axes.plot(funcs[i][j](x))

    return fig, axes

def calc_curvature(trajectory):
    v = calc_vel(trajectory)
    w = calc_omega(trajectory)
    return w/v

def calc_vel(trajectory):
    return np.linalg.norm(trajectory[:, 1, :], axis=0)

def calc_omega(trajectory):
    vx, vy = trajectory[0, 1, :], trajectory[1, 1, :]
    ax, ay = trajectory[0, 2, :], trajectory[1, 2, :]
    v = calc_vel(trajectory)

    return (vx*ay - vy*ax)/v**2

def calc_accel(trajectory):
    vx, vy = trajectory[0, 1, :], trajectory[1, 1, :]
    ax, ay = trajectory[0, 2, :], trajectory[1, 2, :]
    v = calc_vel(trajectory)

    return (vx*ax + vy*ay)/v

def calc_alpha(trajectory):
    vx, vy = trajectory[0, 1, :], trajectory[1, 1, :]
    ax, ay = trajectory[0, 2, :], trajectory[1, 2, :]
    jx, jy = trajectory[0, 3, :], trajectory[1, 3, :]
    v = calc_vel(trajectory)
    a = calc_accel(trajectory)
    k = calc_curvature(trajectory)

    return (vx*jy - vy*jx)/v**2 - 2*a*k


def _unconstrained_from_mat(args, start=None, stop=None, step=None):
    slc = slice(start, stop, step)

    dt = args["dt"][0][0]
    xd_mat = args["xd_mat"]

    xd_mat = xd_mat[:, :, slc]
    dt *= step if step else 1
    return xd_mat, args["Q"], args["R"], args["S"], dt

def test_constrained():
    """Test constrained."""
    xd_mat, q_mat, r_mat, s_mat, a_mat, b_vec, dt = _constrained_from_mat(_load())
    result = smooth_constrained(xd_mat[:, :-1, :], q_mat, r_mat, s_mat, a_mat,
                                b_vec, dt)
    _plot_traj_and_states(result[0], result[1], xd_mat)
    return result

def _constrained_from_mat(args, start=None, stop=None, step=None):
    xd_mat, q_mat, r_mat, s_mat, dt = _unconstrained_from_mat(args, start, stop, step)
    slc = slice(start, stop, step)
    return xd_mat, q_mat, r_mat, s_mat, args["A"][:, :, :, slc], args["b"][:, slc], dt

def test_obstacles_mip():
    """Test MIP Obstacle avoidance."""
    xd_mat, q_mat, r_mat, s_mat, a_mat, b_vec, dt = _constrained_from_mat(_load())
    if CREATE_FREE_REGIONS:
        free_regions = _create_free_regions(np.squeeze(xd_mat[:, 0, :]))
    else:
        free_regions = _free_regions_from(WORLD)
    # axes.plot(xd_mat[0, 0, :].flatten(), xd_mat[1, 0, :].flatten())

    result = smooth_obstacles_mip(xd_mat[:, :-1, :], q_mat, r_mat, s_mat, a_mat,
                                  b_vec, free_regions, dt)
    fig, axes = _plot_traj_and_states(result[0], result[1], xd_mat)
    fig, axes = _plot_free_regions(free_regions, fig=fig, axes=axes)
    return result

def _create_free_regions(position, dist=5.0):
    offset = np.array([[dist, -dist, -dist, dist], [dist, dist, -dist, -dist]])
    curr_pos = position[:, 0].reshape(-1, 1)
    curr_idx = 0
    polys = []
    while curr_idx < position.shape[1]:
        box_vertices = np.transpose(curr_pos+offset)
        polys.append(poly.Polytope.from_vertices_rays(box_vertices))
        tmp_idx, tmp_pos = _get_next_idx_pos_after(position, curr_pos, curr_idx,
                                                   dist - dist/20)
        curr_idx, curr_pos = _get_next_idx_pos_after(position, tmp_pos, tmp_idx,
                                                     dist - dist/20)


    # _plot_free_regions(polys)
    # plt.show()
    return [FreeRegion(*region.get_a_b()) for region in polys]

def _get_next_idx_pos_after(position, curr_pos, curr_idx, dist):
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

def _free_regions_from(regions_dict):
    polys = []
    for _, region in regions_dict.items():
        vertices = np.array(region).T
        polys.append(poly.Polytope.from_vertices_rays(vertices))

    return [FreeRegion(*region.get_a_b()) for region in polys]

def _plot_free_regions(regions, fig=None, axes=None):
    polys = []
    for region in regions:
        if not isinstance(region, poly.Polytope):
            region = poly.Polytope.from_a_b(region.A, region.b)
        polys.append(plt.Polygon(region.get_vertices_rays()[0]))
    poly_collection = plt.matplotlib.collections.PatchCollection(polys,
                                                                 alpha=0.4)
    if not fig or not axes:
        fig, axes = plt.subplots()
    axes.add_collection(poly_collection)
    axes.autoscale()
    return fig, axes

def test_obstacles_short():
    start, stop, step = 1200, 2200, 10
    # start, stop, step = 1600, 2000, 4
    # start, stop, step = 1000, 2000, 10
    # start, stop, step = 1600, 2600, 10
    xd_mat, q_mat, r_mat, s_mat, a_mat, b_vec, dt = _constrained_from_mat(_load(), start, stop, step)
    if CREATE_FREE_REGIONS:
        free_regions = _create_free_regions(np.squeeze(xd_mat[:, 0, :]))
    else:
        free_regions = _free_regions_from(WORLD)
    fig, axes = _plot_free_regions(free_regions)
    axes.plot(xd_mat[0, 0, :].flatten(), xd_mat[1, 0, :].flatten())
    axes.axis("equal")
    plt.show()

    result = smooth_obstacles_mip(xd_mat[:, :-1, :], q_mat, r_mat, s_mat, a_mat,
                                  b_vec, free_regions, dt)
    x = result[0].reshape(xd_mat.shape[0], xd_mat.shape[1]-1, xd_mat.shape[2], order="F")
    # fig, axes = _plot_traj_and_states(result[0], result[1], xd_mat)
    fig, axes = plt.subplots()
    axes.plot(xd_mat[0, 0, :], xd_mat[1, 0, :], x[0, 0, :], x[1, 0, :])
    axes.axis("equal")
    fig, axes = _plot_free_regions(free_regions, fig=fig, axes=axes)
    return result

if __name__ == "__main__":
    import time
    t_start = time.time()
    # test_unconstrained()
    # test_constrained()
    # test_obstacles_mip()
    test_obstacles_short()
    print(time.time()-t_start)
    plt.show()
