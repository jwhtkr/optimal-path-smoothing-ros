"""Path smoothing problem as an LP."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.linalg as linalg

import optimal_control.sparse_utils as sparse
import geometry.polytope as poly


def smooth_path_lp(desired_path, q_mat, r_mat, s_mat, a_mat, b_mat, time_step):
    ndim, nint, nstep = desired_path.shape
    x_initial = desired_path[:, :, 0]
    x_final = desired_path[:, :, -1]
    A_eq, b_eq = discrete_dynamic_equalities(ndim, nint, nstep, x_initial,
                                               time_step)
    A_eq, b_eq = add_terminal_constraint(A_eq, b_eq, x_final, nstep)
    A_ub, b_ub = unpack_a_b(a_mat, b_mat, nstep)
    c, A_ub, b_ub, A_eq, b_eq, bounds = add_slack_variables(A_ub, b_ub, A_eq,
                                                            b_eq, desired_path,
                                                            q_mat, r_mat, s_mat,
                                                            ndim, nint, nstep)
    result = opt.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
    path = remove_slack(result.x)

def discrete_dynamic_equalities(ndim, nint, nstep, x_initial, time_step):
    I = sparse.eye(ndim*nint)
    A_bar, B_bar = discrete_dynamics(ndim, nint, time_step)
    AI = sparse.hstack([A_bar, -I])
    A_x = sparse.block_diag([-I if i == 0 else AI for i in range(nstep)])
    A_u = sparse.block_diag([B_bar for i in range(nstep-1)])
    A_u = sparse.vstack([sparse.coo_matrix((ndim*nint, ndim*(nstep-1))), A_u])

    A = sparse.hstack([A_x, A_u])
    b = np.zeros((A.shape[0],))
    b[:ndim*nint] = -x_initial.reshape(-1, 1, order='F')

    return A, b

def discrete_dynamics(ndim, nint, time_step):
    A_c, B_c = continuous_dynamics(ndim, nint)
    M = np.block([[A_c, B_c], [np.zeros((ndim, ndim*nint)), np.eye(ndim)]])
    discrete = linalg.expm(M*time_step)

    A = discrete[:ndim*nint, :ndim*nint]
    B = discrete[:ndim*nint, ndim*nint:]

    return A, B

def continuous_dynamics(ndim, nint):
    A = np.eye(ndim*nint, k=ndim)
    B = np.eye(*(ndim*nint, ndim))
    return A, B

def add_terminal_constraint(A_eq, b_eq, x_final, nstep):
    ndim, nint = x_final.shape
    A_terminal = sparse.hstack([sparse.coo_matrix((ndim*nint, ndim*nint(nstep-1))),
                                sparse.eye(ndim*nint),
                                sparse.coo_matrix((ndim*nint, ndim*(nstep-1)))])
    b_terminal = -x_final.reshape(-1, 1, order='F')

    A = sparse.vstack([A_eq, A_terminal])
    b = np.concatenate([b_eq, b_terminal])

    return A, b

def unpack_a_b(a_mat, b_mat, nstep):
    nconstr = a_mat.shape[0]
    As, Bs = [], []
    for i in range(nstep):
        As.append(a_mat[:, :, :-1, i].reshape(nconstr, -1, order='F'))
        Bs.append(a_mat[:, :, -1, i].reshape(nconstr, -1, order='F'))

    A_x = sparse.block_diag(As)
    A_u = sparse.block_diag(Bs)
    A = sparse.hstack([A_x, A_u])
    b = b_mat.reshape(-1, 1, order='F')

    return A, b

def add_slack_variables(A_ub, b_ub, A_eq, b_eq, desired_path, q_mat, r_mat,
                        s_mat, ndim, nint, nstep):
    nslack = 2*nstep - 1
    A_eq = sparse.hstack([A_eq, sparse.coo_matrix((A_eq.shape[0], nslack))])
    A_ub = sparse.hstack([A_ub, sparse.coo_matrix((A_ub.shape[0], nslack))])

    A_slack, b_slack = slack_constraints(desired_path, q_mat, r_mat, s_mat,
                                         ndim, nint, nstep)

    A_ub = sparse.vstack([A_ub, A_slack])
    b_ub = np.concatenate([b_ub, b_slack])

    c = slack_cost(nslack)
    bounds = [(None, None) for i in range(ndim*nint*nstep+ndim*(nstep-1))]
    bounds.extend([(0, None) for i in range(nslack)])

    return c, A_ub, b_ub, A_eq, b_eq, bounds
