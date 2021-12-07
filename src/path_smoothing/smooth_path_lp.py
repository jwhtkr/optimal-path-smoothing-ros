"""Path smoothing problem as an LP."""
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

import time
import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.linalg as linalg
import gurobipy as gp
import osqp

import optimal_control.sparse_utils as sparse
# import geometry.polytope as poly
from path_smoothing.smooth_traj_opt import (_load, _constrained_from_mat,
                                            _plot_traj_and_states, _calc_big_m,
                                            _create_free_regions,
                                            _free_regions_from,
                                            _plot_free_regions,
                                            CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE,
                                            WORLD_SHORT_N,
                                            WORLD_SHORT)


def smooth_path_qp(desired_path, q_mat, r_mat, s_mat, a_mat, b_mat,
                   free_regions, time_step): # noqa: D103
    start_time = time.time()
    ndim, nint, nstep = desired_path.shape
    nx, nu = ndim*nint*nstep, ndim*(nstep-1)
    x_initial = desired_path[:, :, 0]
    x_final = desired_path[:, :, -1]
    xd = desired_path.flatten(order="F")
    A_eq, b_eq = discrete_dynamic_equalities(ndim, nint, nstep, x_initial,
                                             time_step)
    A_eq, b_eq = add_terminal_constraint(A_eq, b_eq, x_final, nstep)
    A_ub, b_ub = unpack_a_b(a_mat, b_mat, nstep)
    P_x, P_u, q_x, q_u = quadratic_cost(desired_path, q_mat, r_mat, s_mat,
                                        ndim, nint, nstep)
    Ps, qs = (P_x, P_u), (q_x, q_u)
    if free_regions:
        nbin = len(free_regions)
        na = nstep*nbin
        A_eq, b_eq = eq_add_mip(A_eq, b_eq, nbin, na, nstep)
        A_ub, b_ub = ub_add_mip(A_ub, b_ub, free_regions, ndim, nint, nstep)
        P_a, q_a = sparse.coo_matrix((na, na)), np.zeros((na,))
        Ps += (P_a,)
        qs += (q_a,)

    A_eq = A_eq.tocsr()
    A_ub = A_ub.tocsr()
    A_eqs = A_eq[:, :nx], A_eq[:, nx:nx+nu]
    A_ubs = A_ub[:, :nx], A_ub[:, nx:nx+nu]
    if free_regions:
        A_eq_a, A_ub_a = A_eq[:, nx+nu:], A_ub[:, nx+nu:]
        A_eqs += (A_eq_a,)
        A_ubs += (A_ub_a,)
    create_time = time.time()
    print("Create Time: {:.3f}".format(create_time - start_time))
    m = gp.Model()
    # from pdb import set_trace
    # set_trace()
    if not DISPLAY_SOLVER_OUTPUT:
        m.setParam("outputflag", 0)
    # m.setParam("method", 1)
    m.setParam("FeasibilityTol", 1e-4)
    m.setParam("OptimalityTol", 1e-4)
    x = m.addMVar((nx,), lb=-gp.GRB.INFINITY, name="x")
    u = m.addMVar((nu,), lb=-gp.GRB.INFINITY, name="u")
    variables = (x, u)
    if free_regions:
        a = m.addMVar((na,), vtype=gp.GRB.BINARY, name="alpha")
        variables += (a,)

    eq_expr = sum(A @ var for A, var in zip(A_eqs, variables))
    ub_expr = sum(A @ var for A, var in zip(A_ubs, variables))
    quad_expr = sum(var @ P @ var for P, var in zip(Ps, variables))
    lin_expr = sum(q @ var for q, var in zip(qs, variables))
    m.addConstr(eq_expr == b_eq, name="eq")
    m.addConstr(ub_expr <= b_ub, name="ineq")
    m.setObjective(quad_expr + lin_expr, sense=gp.GRB.MINIMIZE)
    gurobi_time = time.time()
    print ("Gurobi Time: {:.3f}".format(gurobi_time - create_time))

    print("Setup Time: {:.3f}".format(time.time() - start_time))
    # print(time.time()-start_time)
    m.optimize()
    # print(time.time() - start_time)
    path = np.concatenate([x.x, u.x])
    objective = m.getAttr("objval") + xd.T @ xd
    runtime = m.getAttr("runtime")
    return path[:nx], path[nx:], objective, runtime

def smooth_path_lp(desired_path, q_mat, r_mat, s_mat, a_mat, b_mat,
                   free_regions, time_step, mode="infinity"): # noqa: D103
    ndim, nint, nstep = desired_path.shape
    nx, nu = ndim*nint*nstep, ndim*(nstep-1)
    x_initial = desired_path[:, :, 0]
    x_final = desired_path[:, :, -1]
    A_eq, b_eq = discrete_dynamic_equalities(ndim, nint, nstep, x_initial,
                                               time_step)
    A_eq, b_eq = add_terminal_constraint(A_eq, b_eq, x_final, nstep)
    A_ub, b_ub = unpack_a_b(a_mat, b_mat, nstep)
    c, A_ub, b_ub, A_eq, b_eq, bounds = add_slack_variables(A_ub, b_ub, A_eq,
                                                            b_eq, desired_path,
                                                            q_mat, r_mat, s_mat,
                                                            ndim, nint, nstep,
                                                            mode)
    ne = c.size - nx - nu
    cs = c[:nx], c[nx:nx+nu], c[nx+nu:nx+nu+ne]
    if free_regions:
        nbin = len(free_regions)
        na = nstep*nbin
        A_eq, b_eq = eq_add_mip(A_eq, b_eq, nbin, na, nstep)
        A_ub, b_ub = ub_add_mip(A_ub, b_ub, free_regions, ndim, nint, nstep)
        c_a = np.zeros((na,))
        cs += (c_a,)

    A_eq = A_eq.tocsr()
    A_ub = A_ub.tocsr()
    A_eqs = A_eq[:, :nx], A_eq[:, nx:nx+nu], A_eq[:, nx+nu:nx+nu+ne]
    A_ubs = A_ub[:, :nx], A_ub[:, nx:nx+nu], A_ub[:, nx+nu:nx+nu+ne]
    if free_regions:
        A_eq_a, A_ub_a = A_eq[:, nx+nu+ne:nx+nu+ne+na], A_ub[:, nx+nu+ne:nx+nu+ne+na]
        A_eqs += (A_eq_a,)
        A_ubs += (A_ub_a,)


    ############ Gurobi ##############
    m = gp.Model()
    if not DISPLAY_SOLVER_OUTPUT:
        m.setParam("outputflag", 0)
    m.setParam("method", 2)
    m.setParam("nodeMethod", 2)
    x = m.addMVar((nx,), lb=-gp.GRB.INFINITY, name="x")
    u = m.addMVar((nu,), lb=-gp.GRB.INFINITY, name="u")
    e = m.addMVar((ne,), name="epsilon")
    variables = (x, u, e)
    if free_regions:
        a = m.addMVar((na,), vtype=gp.GRB.BINARY, name="alpha")
        variables += (a,)
    eq_expr = sum(A @ var for A, var in zip(A_eqs, variables))
    ub_expr = sum(A @ var for A, var in zip(A_ubs, variables))
    obj_expr = sum(c @ var for c, var in zip(cs, variables))
    m.addConstr(eq_expr == b_eq, name="eq")
    m.addConstr(ub_expr <= b_ub, name="ineq")
    m.setObjective(obj_expr, sense=gp.GRB.MINIMIZE)
    m.optimize()
    objective = m.getAttr("objval")
    runtime = m.getAttr("runtime")
    path = np.concatenate([x.x, u.x])

    ############# OSQP ##################
    # P, q, A, l, u = formulate_lp_osqp(c, A_eq, b_eq, A_ub, b_ub, bounds,
    #                                   ndim, nint, nstep, ne)
    # m_osqp = osqp.OSQP()
    # m_osqp.setup(P, q, A, l, u, verbose=DISPLAY)
    # result = m_osqp.solve()
    # objective = result.info.obj_val
    # runtime = result.info.run_time
    # path = remove_slack(result.x, ne)

    ############# Scipy ################   Not set up to give correct return values
    # result = opt.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
    # path = remove_slack(result.x, ne)


    return path[:nx], path[nx:], objective, runtime

def discrete_dynamic_equalities(ndim, nint, nstep, x_initial, time_step): # noqa: D103
    I = sparse.eye(ndim*nint, format='coo')
    A_bar, B_bar = discrete_dynamics(ndim, nint, time_step)
    AI = sparse.hstack([A_bar, -I], format='coo')
    A_x_rows = [sparse.coo_matrix((-I.data, (I.row, I.col)),
                                  shape=(ndim*nint, ndim*nint*nstep))
                if i == 0 else
                sparse.coo_matrix((AI.data, (AI.row, AI.col+((i-1)*ndim*nint))),
                                  shape=(ndim*nint, ndim*nint*nstep))
                for i in range(nstep)]
    A_x = sparse.vstack(A_x_rows)
    A_u = sparse.block_diag([B_bar for i in range(nstep-1)])
    A_u = sparse.vstack([sparse.coo_matrix((ndim*nint, ndim*(nstep-1))), A_u])

    A = sparse.hstack([A_x, A_u])
    b = np.zeros((A.shape[0],))
    b[:ndim*nint] = -x_initial.flatten(order='F')

    return A, b

def discrete_dynamics(ndim, nint, time_step): # noqa: D103
    # A_c, B_c = continuous_dynamics(ndim, nint)
    # M = np.block([[A_c, B_c], [np.zeros((ndim, ndim*nint)), np.eye(ndim)]])
    # discrete = linalg.expm(M*time_step)

    # A = discrete[:ndim*nint, :ndim*nint]
    # B = discrete[:ndim*nint, ndim*nint:]

    I = np.eye(ndim)
    Z = np.zeros((ndim, ndim))
    AB_rows = []
    for i in range(nint):
        row = []
        for j in range(nint+1):
            if j < i:
                row.append(Z)
            else:
                row.append((time_step**(j-i))/math.factorial(j-i) * I)
        AB_rows.append(np.concatenate(row, axis=1))
    AB = np.concatenate(AB_rows, axis=0)

    A, B = AB[:, :ndim*nint], AB[:, ndim*nint:]
    return A, B

def continuous_dynamics(ndim, nint): # noqa: D103
    A = np.eye(ndim*nint, k=ndim)
    B = np.eye(*(ndim*nint, ndim), k=-ndim*(nint-1))
    return A, B

def add_terminal_constraint(A_eq, b_eq, x_final, nstep): # noqa: D103
    ndim, nint = x_final.shape
    A_terminal = sparse.hstack([sparse.coo_matrix((ndim*nint, ndim*nint*(nstep-1))),
                                -sparse.eye(ndim*nint),
                                sparse.coo_matrix((ndim*nint, ndim*(nstep-1)))])
    b_terminal = -x_final.flatten(order='F')

    A = sparse.vstack([A_eq, A_terminal])
    b = np.concatenate([b_eq, b_terminal])

    return A, b

def unpack_a_b(a_mat, b_mat, nstep): # noqa: D103
    if a_mat.shape[0] == 0:
        return sparse.coo_matrix((0, a_mat.shape[1]*a_mat.shape[2]*a_mat.shape[3]-a_mat.shape[1])), np.empty((0,))
    nconstr = a_mat.shape[0]
    As, Bs = [], []
    for i in range(nstep):
        As.append(a_mat[:, :, :-1, i].reshape(nconstr, -1, order='F'))
        if i < nstep-1:
            Bs.append(a_mat[:, :, -1, i].reshape(nconstr, -1, order='F'))
        else:
            Bs.append(np.empty((nconstr, 0)))

    A_x = sparse.block_diag(As)
    A_u = sparse.block_diag(Bs)
    A = sparse.hstack([A_x, A_u])
    b = b_mat.flatten(order='F')

    return A, b

def add_slack_variables(A_ub, b_ub, A_eq, b_eq, desired_path, q_mat, r_mat,
                        s_mat, ndim, nint, nstep, mode): # noqa: D103
    modes = ["infinity", "one"]
    if mode not in modes:
        raise ValueError("LP mode not one of {}".format(modes))
    if mode == modes[0]:
        nslack = 2*nstep - 1
    if mode == modes[1]:
        nslack = (ndim*nint*nstep) + (ndim*(nstep-1))
    A_eq = sparse.hstack([A_eq, sparse.coo_matrix((A_eq.shape[0], nslack))])
    A_ub = sparse.hstack([A_ub, sparse.coo_matrix((A_ub.shape[0], nslack))])

    A_slack, b_slack = slack_constraints(desired_path, q_mat, r_mat, s_mat,
                                         ndim, nint, nstep, nslack, mode)

    A_ub = sparse.vstack([A_ub, A_slack])
    b_ub = np.concatenate([b_ub, b_slack])

    c = slack_cost(ndim, nint, nstep, nslack)
    bounds = [(None, None) for i in range(ndim*nint*nstep+ndim*(nstep-1))]
    bounds.extend([(0, None) for i in range(nslack)])

    return c, A_ub, b_ub, A_eq, b_eq, bounds

def slack_constraints(desired_path, q_mat, r_mat, s_mat, ndim, nint, nstep, nslack, mode): # noqa: D103
    q_minus_q = np.concatenate([q_mat, -q_mat], axis=0)
    s_minus_s = np.concatenate([s_mat, -s_mat], axis=0)
    r_minus_r = np.concatenate([r_mat, -r_mat], axis=0)
    if mode == "infinity":
        ones_x = np.ones((ndim*nint*2,1))
        ones_u = np.ones((ndim*2,1))
    if mode == "one":
        ones_x = sparse.vstack([sparse.eye(ndim*nint) for _ in range(2)])
        ones_u = sparse.vstack([sparse.eye(ndim) for _ in range(2)])

    A_sub_x = sparse.block_diag([q_minus_q if i < nstep-1 else s_minus_s
                                 for i in range(nstep)])
    A_sub_u = sparse.block_diag([r_minus_r for i in range(nstep-1)])
    e_x = sparse.block_diag([-ones_x for i in range(nstep)])
    e_u = sparse.block_diag([-ones_u for i in range(nstep-1)])

    A_x_u = sparse.block_diag([A_sub_x, A_sub_u])
    A_e = sparse.block_diag([e_x, e_u])

    A = sparse.hstack([A_x_u, A_e])
    b = A @ np.concatenate([desired_path.flatten(order='F'),
                            np.zeros((ndim*(nstep-1),)),
                            np.zeros((nslack,))])

    return A, b

def slack_cost(ndim, nint, nstep, nslack): # noqa: D103
    return np.concatenate([np.zeros((ndim*nint*nstep,)),
                           np.zeros((ndim*(nstep-1),)),
                           np.ones((nslack,))])

def remove_slack(x, nslack): # noqa: D103
    return x[:-nslack]

def formulate_lp_osqp(c, A_eq, b_eq, A_ub, b_ub, bounds, ndim, nint, nstep, nslack):
    bounds = np.asarray(bounds)
    n = ndim*nint*nstep + ndim*(nstep-1) + nslack
    P = sparse.csc_matrix((n, n))
    q = c
    A = sparse.vstack([A_eq, A_ub,
                       sparse.eye(nslack, n, k=ndim*(nstep*(nint+1)-1))],
                      format="csc")
    l = np.concatenate([b_eq, -np.inf*np.ones((A_ub.shape[0],)), np.zeros((nslack,))])
    u = np.concatenate([b_eq, b_ub, np.inf*np.ones((nslack,))])

    return P, q, A, l, u

def quadratic_cost(xd_mat, q_mat, r_mat, s_mat, ndim, nint, nstep):
    xd = xd_mat.flatten(order="F")
    P_x = sparse.block_diag([q_mat if i < nstep-1 else s_mat
                             for i in range(nstep)])
    q_x = -2 * xd @ P_x

    P_u = sparse.block_diag([r_mat for i in range(nstep-1)])
    q_u = sparse.csr_matrix((1, ndim*(nstep-1)))

    return P_x, P_u, q_x, q_u

def eq_add_mip(A_eq, b_eq, nbin, na, nstep):
    A_sum_a = sparse.block_diag([np.ones((nbin,)) for _ in range(nstep)])
    A_sum_rest = sparse.coo_matrix((A_sum_a.shape[0], A_eq.shape[1]))
    A_sum = sparse.hstack([A_sum_rest, A_sum_a])
    b_sum = nbin*np.ones((nstep,)) - 1

    A = sparse.hstack([A_eq, sparse.coo_matrix((A_eq.shape[0], na))])
    A = sparse.vstack([A, A_sum])
    b = np.concatenate([b_eq, b_sum])
    return A, b

def ub_add_mip(A_ub, b_ub, free_regions, ndim, nint, nstep):
    if len(free_regions) > 1:
        As = [sparse.hstack([A, sparse.coo_matrix((A.shape[0], ndim*(nint-1)))])
          for A, _ in free_regions]
        Ms = [-_calc_big_m(region, np.array([[-50, 50], [-50, 50]]))
          for region in free_regions]
        A_rs = sparse.vstack(As)
        b_rs = np.concatenate([b for _, b in free_regions])
        M_rs = sparse.block_diag(Ms)
    else:
        A_r, b_r = free_regions[0]
        A_r = np.concatenate([A_r, np.zeros((A_r.shape[0], ndim*(nint-1)))], axis=1)
        M_r = -_calc_big_m(free_regions[0], np.array([[-50, 50], [-50, 50]]))

        b_rs = b_r
        A_rs = sparse.coo_matrix(A_r)
        M_rs = sparse.coo_matrix(M_r)

    A_x_obs = sparse.block_diag([A_rs for _ in range(nstep)])
    A_rest_obs = sparse.coo_matrix((A_x_obs.shape[0], A_ub.shape[1]-A_x_obs.shape[1]))
    A_a_obs = sparse.block_diag([M_rs for _ in range(nstep)])
    A_obs = sparse.hstack([A_x_obs, A_rest_obs, A_a_obs])
    b_obs = np.concatenate([b_rs for _ in range(nstep)])

    A = sparse.hstack([A_ub,
                       sparse.coo_matrix((A_ub.shape[0], A_a_obs.shape[1]))])
    A = sparse.vstack([A, A_obs])
    b = np.concatenate([b_ub, b_obs])

    return A, b

def _slice(xd_mat, a_mat, b_mat, dt, step):
    return xd_mat[:, :, ::step], a_mat[:, :, :, ::step], b_mat[:, ::step], dt*step

def _shorten(xd_mat, a_mat, b_mat, dt, nsteps, step):
    start = 850
    end = start + nsteps*step
    return xd_mat[:, :, start:end:step], a_mat[:, :, :, start:end:step], b_mat[:, start:end:step], dt*step

def _calc_slice(num, stop, *, start=0):
    nspan = stop - start
    eps = nspan/num/100
    out = np.empty_like(num, dtype=np.int)
    step = np.floor(nspan/num + eps, out=out, casting="unsafe")
    # step = math.floor(nspan/num + eps)
    # calculate correct stop value to get num
    stop = (num - 1)*step + start + 1
    return start, stop, step

def _calc_short(num, step):
    end = (num - 1)*step
    start = 1200 if end < 2200 else 0
    stop = (num - 1)*step + start + 1
    return start, stop, step

def _mip_args(start, stop, step, world, suffix):
    args = list(_constrained_from_mat(_load(), start, stop, step))
    args.insert(6, _free_regions_from(world))
    args.append(suffix)
    return args

def _display(results, name, xd_mat):
    print("Solve Time: {:.3f}".format(results[3]))
    if PLOT:
        fig, axes = _plot_traj_and_states(results[0], results[1], xd_mat)
        fig.suptitle("{} - Time: {:.3f}\nObj. Value: {:.3f}".format(name, results[3], results[2]))
        return fig, axes
    return None, None

def _results(xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, free_regions, dt, suffix=""):
    if suffix:
        suffix = "-" + suffix

    tests = {"LP-inf": (smooth_path_lp, "LP-$\\infty$"),
             "LP-one": (lambda *args: smooth_path_lp(*args, "one"), "LP-1"),
             "QP": (smooth_path_qp, "QP")}
    # tests = {"QP": (smooth_path_qp, "QP")}

    figs = []

    for k, v in tests.items():
        name = k + suffix
        fig_name = v[1] + suffix
        print("--------------------{}--------------------".format(name))
        result = v[0](xd_mat[:, :-1, :], q_mat, r_mat, s_mat, a_mat, b_mat, free_regions, dt)
        fig, axes = _display(result, fig_name, xd_mat)
        if free_regions:
            _plot_free_regions(free_regions, fig=fig, axes=axes)
        if fig:
            figs.append(fig)

    return figs

def _extract_times(figs):
    times = []
    for fig in figs:
        title = fig.texts[0].get_text()
        line1 = title.splitlines()[0]
        time_text = line1.split(" - ")[-1]
        time = float(time_text[5:])
        times.append(time)

    return times

def test_simple():
    # xd_mat = np.array([[[0, 1, 2], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    xd = np.concatenate([np.linspace(0, 10), np.full((50,), 10)])
    yd = np.concatenate([np.zeros((50,)), np.linspace(0, 10)])
    xd_vec = np.concatenate([xd.reshape(1, -1), yd.reshape(1, -1), np.zeros((8, 100))], axis=0)
    xd_mat = xd_vec.reshape((2, 5, -1), order='F')
    xd_mat[:, 1, :50] = np.array([1, 0]).reshape(-1, 1)
    xd_mat[:, 1, -50:] = np.array([0, 1]).reshape(-1, 1)

    ndim, nint, nstep = xd_mat.shape
    q_mat = np.diag([1, 1, 0, 0, 10, 10, 10, 10])
    r_mat = np.diag([1, 1])
    s_mat = q_mat
    a_mat = np.empty((0, ndim, nint, nstep))
    b_mat = np.empty((0, nstep))
    dt = 0.2

    world = {1: ((0., 10., 10., 7.),
                 (0., 10., 3., 0.))}

    free_regions = _free_regions_from(world)

    return _results(xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, free_regions, dt, "simple")

def test_full():
    args = list(_constrained_from_mat(_load()))
    args.insert(6, _free_regions_from(CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE))
    return _results(*args)  # pylint: disable=no-value-for-parameter

def test_sliced(n):
    xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, dt = _constrained_from_mat(_load())
    xd_mat_sliced, _, _, _, a_mat_sliced, b_mat_sliced, dt_sliced = _constrained_from_mat(_load(), *_calc_slice(n, xd_mat.shape[2]))
    # q_mat = np.diag([1, 1, 0, 0, 1, 1, 1, 1])
    # r_mat = np.diag([10, 10])
    free_regions = _free_regions_from(CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE)

    return _results(xd_mat_sliced, q_mat, r_mat, s_mat, a_mat_sliced,
                    b_mat_sliced, free_regions, dt_sliced,
                    "sliced (N={}, dt={:.2f})".format(n, dt_sliced))

def test_short(n):
    xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, dt = _constrained_from_mat(_load())
    xd_mat_short, _, _, _, a_mat_short, b_mat_short, dt_short = _constrained_from_mat(_load(), *_calc_short(n, 1000//n))
    # q_mat = np.diag([1, 1, 0, 0, 1, 1, 1, 1])
    # r_mat = np.diag([10, 10])
    # free_regions = _free_regions_from(CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE)
    free_regions = _free_regions_from(WORLD_SHORT_N[2])

    return _results(xd_mat_short, q_mat, r_mat, s_mat, a_mat_short, b_mat_short,
                    free_regions, dt_short,
                    "short (N={}, dt={:.2f})".format(n, dt_short))

def optimization_time():
    xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, dt = _constrained_from_mat(_load())
    nstep = xd_mat.shape[2]
    ns = np.logspace(1, 4, num=25, endpoint=False, dtype=np.int)
    # ns = ns[ns < 1000]
    ns = ns[ns < nstep]
    ns = np.concatenate([ns, [nstep]])

    cases = ["LP-$\\infty$", "LP-one", "QP"]
    case_funcs = [smooth_path_lp, lambda *args: smooth_path_lp(*args, "one"),
                  smooth_path_qp]

    result_arr = np.ma.empty((2*len(cases), len(ns), 4), dtype=np.object)

    start_time = time.time()
    for j, n in enumerate(ns):
        print("-------------{}/{}-------------".format(j+1, len(ns)))
        sliced_args = list(_constrained_from_mat(_load(), *_calc_slice(n, nstep)))
        short_args = list(_constrained_from_mat(_load(), *_calc_short(n, 1)))
        sliced_args[0] = sliced_args[0][:, :-1, :]
        short_args[0] = short_args[0][:, :-1, :]
        # sliced_args.insert(6, [])
        # short_args.insert(6, [])
        sliced_args.insert(6, _free_regions_from(CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE))
        short_args.insert(6, _free_regions_from(CORRIDOR_WORLD_STRAIGHT_WITH_OBSTACLE))
        for i, _ in enumerate(cases):
            row = 2*i
            try:
                result_arr[row, j] = case_funcs[i](*sliced_args)
                print("Solve Time: {}".format(result_arr[row, j, 3]))
            except (gp.GurobiError, AttributeError):
                result_arr[row, j] = np.ma.masked
            print("Cumulative Time: {}".format(time.time() - start_time))
            try:
                result_arr[row+1, j] = case_funcs[i](*short_args)
                print("Solve Time: {}".format(result_arr[row+1, j, 3]))
            except (gp.GurobiError, AttributeError):
                result_arr[row+1, j] = np.ma.masked
            print("Cumulative Time: {}".format(time.time() - start_time))

    ### Plot stuff
    fig_time, axes_time = plt.subplots()
    axes_time.semilogx(ns, result_arr[:, :, 3].filled(0).T, "x-")
    axes_time.set_xlabel("Problem size (N)")
    axes_time.set_ylabel("Optimization Solve Time (sec)")
    axes_time.legend([cases[0]+"-sliced", cases[0]+"-short",
                 cases[1]+"-sliced", cases[1]+"-short",
                 cases[2]+"-sliced", cases[2]+"-short"])
    fig_time.suptitle("Solve Time for Various Problem Sizes")

    fig_cost, axes_cost = plt.subplots()
    _, _, steps = _calc_slice(ns, xd_mat.shape[2])
    # axes_cost.plot(steps*0.01, result_arr[::2, :, 2].T*steps[:, np.newaxis]*0.01, "x-")
    # axes_cost.semilogx(steps*0.01, result_arr[::2, :, 2].T*steps[:, np.newaxis]*0.01, "x-")
    axes_cost.semilogx(steps*0.01, result_arr[::2, :, 2].filled(0).T/ns[:, np.newaxis], "x-")
    axes_cost.invert_xaxis()
    axes_cost.set_xlabel("Discretization Resolution $\\Delta t$ (sec)")
    # axes_cost.set_ylabel("Normalized Objective Cost (Cost*$\\Delta t$)")
    axes_cost.set_ylabel("Normalized Objective Cost (Cost/N)")
    axes_cost.legend([cases[0], cases[1], cases[2]])
    fig_cost.suptitle("Optimal Cost for Various Resolutions")

    return [fig_time, fig_cost]

def mip_tests():
    l_cases = [25, 50, 100]
    place_cases = ["mid", "mid-short", "front", "end"]
    place_case_start_stop = [(1200, 2200), (1600, 2000), (1600, 2600), (1000, 2000)]
    n_cases = [1, 2, 3]

    fig_list = []
    fig_places, axes_places = plt.subplots(len(place_cases), 1)
    fig_places.suptitle("Obstacle Placement")
    fig_places.set_tight_layout(True)
    fig_n, axes_n = plt.subplots(len(n_cases), 1)
    fig_n.suptitle("Number of Regions")
    fig_n.set_tight_layout(True)

    times_places = np.zeros((len(place_cases), len(l_cases), 3))
    times_n = np.zeros((len(n_cases), len(l_cases), 3))

    for i, (place, (start, stop)) in enumerate(zip(place_cases, place_case_start_stop)):
        for j, l in enumerate(l_cases):
            args = _mip_args(*_calc_slice(l, stop=stop, start=start),
                             WORLD_SHORT_N[2], "{} (N={}, dt={})")
            args[-1] = args[-1].format(place, l, args[-2])
            figs = _results(*args)  # pylint: disable=no-value-for-parameter
            times_places[i, j, :] = _extract_times(figs)
            fig_list.extend(figs)

        axes_places[i].semilogx(l_cases, times_places[i, :, :])
        axes_places[i].set_title("Obst. placement: {}".format(place))
        axes_places[i].set_ylabel("Time (sec)")
        if i == len(place_cases) - 1:
            axes_places[i].legend(["LP-$\\infty$", "LP-1", "QP"])
            axes_places[i].set_xlabel("Traj. length (N)")

    for i, nregions in enumerate(n_cases):
        for j, l in enumerate(l_cases):
            args = _mip_args(*_calc_short(l, 1000//l), WORLD_SHORT_N[nregions],
                             "{} regions (N={}, dt={})")
            args[-1] = args[-1].format(nregions, l, args[-2])
            figs = _results(*args)  # pylint: disable=no-value-for-parameter
            times_n[i, j, :] = _extract_times(figs)
            fig_list.extend(figs)

        axes_n[i].semilogx(l_cases, times_n[i, :, :])
        axes_n[i].set_title("{} regions".format(nregions))
        axes_n[i].set_ylabel("Time (sec)")
        if i == len(n_cases) - 1:
            axes_n[i].legend(["LP-$\\infty$", "LP-1", "QP"])
            axes_n[i].set_xlabel("Traj. length (N)")

    fig_list.extend([fig_places, fig_n])
    # return fig_list
    return [fig_places, fig_n]


DISPLAY_SOLVER_OUTPUT = False

if __name__ == "__main__":
    PLOT = True
    SAVE_FIGS = False

    n = 50

    fig_list = []

    # start_time = time.time()
    fig_list.extend(test_simple())
    # print("Time: {:.3f}".format(time.time() - start_time))
    # fig_list.extend(test_full())
    # fig_list.extend(test_sliced(n))
    # fig_list.extend(test_short(n))

    # fig_list.extend(optimization_time())

    # fig_list.extend(mip_tests())

    if PLOT and SAVE_FIGS:
        for i, fig in enumerate(fig_list):
            file_type = "svg"
            if fig.texts:
                title = fig.texts[0].get_text()
                top_line = title.splitlines()[0]
                wo_time = top_line.split(" - ")[0]
                wo_time = wo_time.replace("$\\infty$", "inf")
                wo_time = wo_time.replace(".", "_")
                base_name = wo_time
            else:
                base_name = "figure_{}".format(i)

            file_name = ".".join([base_name, file_type])

            fig.savefig(file_name)
    if PLOT:
        plt.show()
        pass
