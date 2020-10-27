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
from path_smoothing.smooth_traj_opt import _load, _constrained_from_mat, _plot_traj_and_states

DISPLAY = False


def smooth_path_qp(desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, time_step): # noqa: D103
    ndim, nint, nstep = desired_path.shape
    nx, nu = ndim*nint*nstep, ndim*(nstep-1)
    x_initial = desired_path[:, :, 0]
    x_final = desired_path[:, :, -1]
    A_eq, b_eq = discrete_dynamic_equalities(ndim, nint, nstep, x_initial,
                                               time_step)
    A_eq, b_eq = add_terminal_constraint(A_eq, b_eq, x_final, nstep)
    A_ub, b_ub = unpack_a_b(a_mat, b_mat, nstep)
    P_x, P_u, q_x, q_u = quadratic_cost(desired_path, q_mat, r_mat, s_mat,
                                        ndim, nint, nstep)

    A_eq = A_eq.tocsr()
    A_ub = A_ub.tocsr()
    A_eq_x, A_eq_u = A_eq[:, :nx], A_eq[:, nx:nx+nu]
    A_ub_x, A_ub_u = A_ub[:, :nx], A_ub[:, nx:nx+nu]

    m = gp.Model()
    if not DISPLAY:
        m.setParam("outputflag", 0)
    m.Params.method = 1
    x = m.addMVar((nx,), lb=-gp.GRB.INFINITY, name="x")
    u = m.addMVar((nu,), lb=-gp.GRB.INFINITY, name="u")
    m.addConstr(A_eq_x @ x + A_eq_u @ u == b_eq, name="eq")
    m.addConstr(A_ub_x @ x + A_ub_u @ u <= b_ub, name="ineq")
    m.setObjective(x @ P_x @ x + u @ P_u @ u + q_x @ x + q_u @ u,
                   sense=gp.GRB.MINIMIZE)

    # print(time.time()-start_time)
    m.optimize()
    # print(time.time() - start_time)
    path = np.concatenate([x.x, u.x])
    runtime = m.runtime
    return path[:nx], path[nx:], runtime

def smooth_path_lp(desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, time_step, mode="infinity"): # noqa: D103
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
    m.Params.method = 2
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
    runtime = m.runtime
    path = np.concatenate([x.x, u.x])

    ############# OSQP ##################
    # P, q, A, l, u = formulate_lp_osqp(c, A_eq, b_eq, A_ub, b_ub, bounds,
    #                                   ndim, nint, nstep, ne)
    # m_osqp = osqp.OSQP()
    # m_osqp.setup(P, q, A, l, u, verbose=DISPLAY)
    # result = m_osqp.solve()
    # runtime = result.info.run_time
    # path = remove_slack(result.x, ne)

    ############# Scipy ################
    # result = opt.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
    # result = opt.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
    # path = remove_slack(result.x, ne)


    return path[:nx], path[nx:], runtime

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
    step = math.floor(nspan/num + eps)
    return start, stop, step

def _calc_short(num, step):
    end = (num - 1)*step
    start = 850 if end < 1000 else 0
    stop = (num - 1)*step + start + 1
    return start, stop, step

PLOT = True
SAVE_FIGS = True

if __name__ == "__main__":
    # xd_mat = np.array([[[0, 1, 2], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    # xd = np.concatenate([np.linspace(0, 10), np.full((50,), 10)])
    # yd = np.concatenate([np.zeros((50,)), np.linspace(0, 10)])
    # xd_vec = np.concatenate([xd.reshape(1, -1), yd.reshape(1, -1), np.zeros((8, 100))], axis=0)
    # xd_mat = xd_vec.reshape(2, 5, 100, order='F')
    # xd_mat[:, 1, 0] = [1, 0]
    # xd_mat[:, 1, -1] = [0, 1]

    # ndim, nint, nstep = xd_mat.shape
    # q_mat = np.diag([1, 1, 0, 0, 10, 10, 10, 10])
    # r_mat = np.diag([1, 1])
    # s_mat = q_mat
    # a_mat = np.empty((0, ndim, nint, nstep))
    # b_mat = np.empty((0, nstep))
    # dt = 0.1

    xd_mat, q_mat, r_mat, s_mat, a_mat, b_mat, dt = _constrained_from_mat(_load())
    # q_mat = np.diag([1, 1, 0, 0, 1, 1, 1, 1])
    # r_mat = np.diag([10, 10])
    n = 100
    xd_mat_sliced, _, _, _, a_mat_sliced, b_mat_sliced, dt_sliced = _constrained_from_mat(_load(), *_calc_slice(n,xd_mat.shape[2]))

    xd_mat_short, _, _, _, a_mat_short, b_mat_short, dt_short = _constrained_from_mat(_load(), *_calc_short(n, 10))
    # start_time = time.time()

    fig_list = []

    print("-----------------LP-inf---------------------")
    lp_inf = smooth_path_lp(xd_mat[:, :-1, :], q_mat, r_mat, s_mat, a_mat, b_mat, dt)
    if PLOT:
        fig_lp_inf, _ = _plot_traj_and_states(lp_inf[0], lp_inf[1], xd_mat)
        fig_lp_inf.suptitle("LP-$\\infty$ - Time: {:.3f}".format(lp_inf[2]))
        fig_list.append(fig_lp_inf)
    else:
        print("Time: {:.3f}".format(lp_inf[2]))

    print("-----------------LP-one---------------------")
    lp_one = smooth_path_lp(xd_mat[:, :-1, :], q_mat, r_mat, s_mat, a_mat, b_mat, dt, "one")
    if PLOT:
        fig_lp_one, _ = _plot_traj_and_states(lp_one[0], lp_one[1], xd_mat)
        fig_lp_one.suptitle("LP-1 - Time: {:.3f}".format(lp_one[2]))
        fig_list.append(fig_lp_one)
    else:
        print("Time: {:.3f}".format(lp_one[2]))

    print("-------------------QP-----------------------")
    qp = smooth_path_qp(xd_mat[:, :-1, :], q_mat, r_mat, s_mat, a_mat, b_mat, dt)
    if PLOT:
        fig_qp, _ = _plot_traj_and_states(qp[0], qp[1], xd_mat)
        fig_qp.suptitle("QP - Time: {:.3f}".format(qp[2]))
        fig_list.append(fig_qp)
    else:
        print("Time: {:.3f}".format(qp[2]))


    print("--------------LP-inf-sliced-----------------")
    lp_inf_sliced = smooth_path_lp(xd_mat_sliced[:, :-1, :], q_mat, r_mat,
                                   s_mat, a_mat_sliced, b_mat_sliced, dt_sliced)
    if PLOT:
        fig_lp_inf_sliced, _ = _plot_traj_and_states(lp_inf_sliced[0],
                                                    lp_inf_sliced[1],
                                                    xd_mat_sliced)
        fig_lp_inf_sliced.suptitle("LP-$\\infty$ - sliced - Time: {:.3f}".format(lp_inf_sliced[2]))
        fig_list.append(fig_lp_inf_sliced)
    else:
        print("Time: {:.3f}".format(lp_inf_sliced[2]))

    print("--------------LP-one-sliced-----------------")
    lp_one_sliced = smooth_path_lp(xd_mat_sliced[:, :-1, :], q_mat, r_mat,
                                   s_mat, a_mat_sliced, b_mat_sliced, dt_sliced,
                                   "one")
    if PLOT:
        fig_lp_one_sliced, _ = _plot_traj_and_states(lp_one_sliced[0],
                                                    lp_one_sliced[1],
                                                    xd_mat_sliced)
        fig_lp_one_sliced.suptitle("LP-1 - sliced - Time: {:.3f}".format(lp_one_sliced[2]))
        fig_list.append(fig_lp_one_sliced)
    else:
        print("Time: {:.3f}".format(lp_one_sliced[2]))

    print("----------------QP-sliced-------------------")
    qp_sliced = smooth_path_qp(xd_mat_sliced[:, :-1, :], q_mat, r_mat, s_mat,
                               a_mat_sliced, b_mat_sliced, dt_sliced)
    if PLOT:
        fig_qp_sliced, _ = _plot_traj_and_states(qp_sliced[0], qp_sliced[1],
                                                xd_mat_sliced)
        fig_qp_sliced.suptitle("QP - sliced - Time: {:.3f}".format(qp_sliced[2]))
        fig_list.append(fig_qp_sliced)
    else:
        print("Time: {:.3f}".format(qp_sliced[2]))


    print("--------------LP-inf-short------------------")
    lp_inf_short = smooth_path_lp(xd_mat_short[:, :-1, :], q_mat, r_mat, s_mat,
                                  a_mat_short, b_mat_short, dt_short)
    if PLOT:
        fig_lp_inf_short, _ = _plot_traj_and_states(lp_inf_short[0],
                                                    lp_inf_short[1],
                                                    xd_mat_short)
        fig_lp_inf_short.suptitle("LP-$\\infty$ - short - Time: {:.3f}".format(lp_inf_short[2]))
        fig_list.append(fig_lp_inf_short)
    else:
        print("Time: {:.3f}".format(lp_inf_short[2]))

    print("--------------LP-one-short------------------")
    lp_one_short = smooth_path_lp(xd_mat_short[:, :-1, :], q_mat, r_mat, s_mat,
                                  a_mat_short, b_mat_short, dt_short, "one")
    if PLOT:
        fig_lp_one_short, _ = _plot_traj_and_states(lp_one_short[0],
                                                    lp_one_short[1],
                                                    xd_mat_short)
        fig_lp_one_short.suptitle("LP-1 - short - Time: {:.3f}".format(lp_one_short[2]))
        fig_list.append(fig_lp_one_short)
    else:
        print("Time: {:.3f}".format(lp_one_short[2]))

    print("----------------QP-short--------------------")
    qp_short = smooth_path_qp(xd_mat_short[:, :-1, :], q_mat, r_mat, s_mat,
                              a_mat_short, b_mat_short, dt_short)
    if PLOT:
        fig_qp_short, _ = _plot_traj_and_states(qp_short[0], qp_short[1],
                                                xd_mat_short)
        fig_qp_short.suptitle("QP - short - Time: {:.3f}".format(qp_short[2]))
        fig_list.append(fig_qp_short)
    else:
        print("Time: {:.3f}".format(qp_short[2]))

    if PLOT and SAVE_FIGS:
        for i, fig in enumerate(fig_list):
            fig.savefig("figure_{}.png".format(i))
    if PLOT and not SAVE_FIGS:
        plt.show()
