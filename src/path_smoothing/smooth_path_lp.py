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
from path_smoothing.smooth_traj_opt import _load, _constrained_from_mat, _plot_traj_and_states, _calc_big_m

DISPLAY = False


def smooth_path_qp(desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, free_regions, time_step): # noqa: D103
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
        nbin = sum(len(b) for _, b in free_regions)
        na = nstep*nbin
        # A_eq, b_eq = eq_add_mip(A_eq, b_eq, free_regions, nstep)
        # A_ub, b_ub = ub_add_mip(A_ub, b_ub, free_regions, nstep)
        P_a, q_a = sparse.coo_matrix((na, na)), sparse.coo_matrix((na,))
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

    m = gp.Model()
    if not DISPLAY:
        m.setParam("outputflag", 0)
    m.setParam("method", 1)
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

    ############ Gurobi ##############
    m = gp.Model()
    if not DISPLAY:
        m.setParam("outputflag", 0)
    m.setParam("method", 2)
    x = m.addMVar((nx,), lb=-gp.GRB.INFINITY, name="x")
    u = m.addMVar((nu,), lb=-gp.GRB.INFINITY, name="u")
    e = m.addMVar((ne,), name="epsilon")
    A_eq = A_eq.tocsr()
    A_ub = A_ub.tocsr()
    A_eq_x, A_eq_u, A_eq_e = A_eq[:, :nx], A_eq[:, nx:nx+nu], A_eq[:, nx+nu:]
    A_ub_x, A_ub_u, A_ub_e = A_ub[:, :nx], A_ub[:, nx:nx+nu], A_ub[:, nx+nu:]
    c_x, c_u, c_e = c[:nx], c[nx:nx+nu], c[nx+nu:]
    m.addConstr(A_eq_x @ x + A_eq_u @ u + A_eq_e @ e == b_eq, name="eq")
    m.addConstr(A_ub_x @ x + A_ub_u @ u + A_ub_e @ e <= b_ub, name="ineq")
    m.setObjective(c_x @ x + c_u @ u + c_e @ e, sense=gp.GRB.MINIMIZE)
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
    start = 850 if end < 1000 else 0
    stop = (num - 1)*step + start + 1
    return start, stop, step

def _display(results, name, xd_mat):
    print("Time: {:.3f}".format(results[3]))
    if PLOT:
        fig, _ = _plot_traj_and_states(results[0], results[1], xd_mat)
        fig.suptitle("{} - Time: {:.3f}\nObj. Value: {:.3f}".format(name, results[3], results[2]))
        return fig

def _results(xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, dt, suffix=""):
    if suffix:
        suffix = "-" + suffix

    tests = {"LP-inf": (smooth_path_lp, "LP-$\\infty$"),
             "LP-one": (lambda *args: smooth_path_lp(*args, "one"), "LP-1"),
             "QP": (smooth_path_qp, "QP")}

    figs = []

    for k, v in tests.items():
        name = k + suffix
        fig_name = v[1] + suffix
        print("--------------------{}--------------------".format(name))
        result = v[0](xd_mat[:, :-1, :], q_mat, r_mat, s_mat, a_mat, b_mat, [], dt)
        fig = _display(result, fig_name, xd_mat)
        if fig:
            figs.append(fig)

    return figs

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

    return _results(xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, dt, "simple")

def test_full():
    return _results(*_constrained_from_mat(_load()))

def test_sliced(n):
    xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, dt = _constrained_from_mat(_load())
    xd_mat_sliced, _, _, _, a_mat_sliced, b_mat_sliced, dt_sliced = _constrained_from_mat(_load(), *_calc_slice(n, xd_mat.shape[2]))
    # q_mat = np.diag([1, 1, 0, 0, 1, 1, 1, 1])
    # r_mat = np.diag([10, 10])

    return _results(xd_mat_sliced, q_mat, r_mat, s_mat, a_mat_sliced, b_mat_sliced, dt_sliced, "sliced")

def test_short(n):
    xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, dt = _constrained_from_mat(_load())
    xd_mat_short, _, _, _, a_mat_short, b_mat_short, dt_short = _constrained_from_mat(_load(), *_calc_short(n, 1))
    # q_mat = np.diag([1, 1, 0, 0, 1, 1, 1, 1])
    # r_mat = np.diag([10, 10])

    return _results(xd_mat_short, q_mat, r_mat, s_mat, a_mat_short, b_mat_short, dt_short, "short")

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
        sliced_args.insert(6, [])
        short_args.insert(6, [])
        for i, _ in enumerate(cases):
            row = 2*i
            try:
                result_arr[row, j] = case_funcs[i](*sliced_args)
                print("Solve Time: {}".format(result_arr[row, j, 3]))
            except gp.GurobiError:
                result_arr[row, j] = np.ma.masked
            print("Cumulative Time: {}".format(time.time() - start_time))
            try:
                result_arr[row+1, j] = case_funcs[i](*short_args)
                print("Solve Time: {}".format(result_arr[row+1, j, 3]))
            except gp.GurobiError:
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
    axes_cost.semilogx(steps*0.01, result_arr[::2, :, 2].T*steps[:, np.newaxis]*0.01, "x-")
    axes_cost.invert_xaxis()
    axes_cost.set_xlabel("Discretization Resolution $\\Delta t$ (sec)")
    axes_cost.set_ylabel("Normalized Objective Cost (Cost*$\\Delta t$)")
    axes_cost.legend([cases[0], cases[1], cases[2]])
    fig_cost.suptitle("Optimal Cost for Various Resoulutions")

    return [fig_time, fig_cost]

if __name__ == "__main__":
    PLOT = False
    SAVE_FIGS = False

    n = 100

    fig_list = []

    fig_list.extend(test_simple())
    # fig_list.extend(test_full())
    # fig_list.extend(test_sliced(n))
    # fig_list.extend(test_short(n))

    # fig_list.extend(optimization_time())

    if PLOT and SAVE_FIGS:
        for i, fig in enumerate(fig_list):
            fig.savefig("figure_{}.svg".format(i))
    if PLOT:
        plt.show()
