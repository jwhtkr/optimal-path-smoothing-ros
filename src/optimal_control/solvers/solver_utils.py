"""Utility functions commonly used for solvers."""


import numpy as np

import optimal_control.sparse_utils as sparse
from optimal_control.objectives import quadratic_cost


def to_condensed(cost):
    """
    Convert a quadratic cost to a `DiscreteCondensedQuadraticCost` object.

    Parameters
    ----------
    cost : quadratic_cost.QuadraticCost
        A quadratic cost to convert to a discrete, condensed quadratic cost.

    Returns
    -------
    quadratic_cost.DiscreteCondensedQuadraticCost
        The cost converted to a `DiscreteCondensedQuadraticCost` form.

    """
    isc = cost.inst_state_cost
    icc = cost.inst_ctrl_cost
    tsc = cost.term_state_cost
    dstate = cost.desired_state
    dctrl = cost.desired_ctrl

    return quadratic_cost.DiscreteCondensedQuadraticCost(isc, icc, tsc, dstate,
                                                         dctrl)


def to_p_q(cost):
    """
    Convert the cost to the P matrix and q vector of the OSQP formulation.

    It is assumed that the cost is already of a form such that the "state"
    and "ctrl" (and their respective desired values) are aggregated. In
    other words, the cost is already converted from a multi-stage discrete
    optimization cost to a parameter optimization cost.

    Parameters
    ----------
    cost : quadratic_cost.DiscreteCondensedQuadraticCost
        The cost to convert to the P matrix and the q vector.

    Returns
    -------
    p_mat : scipy.sparse.csc_matrix
        The P matrix of a quadratic cost of form J(y) = y^T P y + 2 q^T y.
    q_vec : numpy.ndarray
        The q vector of a quadratic cost of form J(y) = y^T P y + 2 q^T y.

    """
    q_vec = sparse.coo_matrix(cost.inst_state_cost)
    r_mat = sparse.coo_matrix(cost.inst_ctrl_cost)
    s_mat = sparse.coo_matrix(cost.term_state_cost)
    qs_mat = q_vec + s_mat
    x_d = cost.desired_state(None)
    u_d = cost.desired_ctrl(None)

    # P = sparse.bmat([[QS, None], [None, R]])
    p_mat = sparse.block_diag([qs_mat, r_mat])
    q_vec = np.concatenate([-qs_mat.transpose().dot(x_d),
                        -r_mat.transpose().dot(u_d)])

    return p_mat.tocsc(), q_vec


def to_a_l_u(constraints):
    """
    Convert the `constraints` object to A, l, and u of the OSQP formulation.

    It is assumed that the constraints in the `constraints` object are
    in their aggregated form. In other words, they are already converted
    from a multi-stage discrete formulation to a parameter optimization
    formulation.

    Parameters
    ----------
    constraints : linear_constraints.LinearConstraints
        The constraints to convert to the OSQP formulation.

    Returns
    -------
    a_mat : scipy.sparse.csc_matrix
        The A matrix of the linear inequality constraints: l <= Ay <= u.
    lb_vec : numpy.ndarray
        The l vector of the linear inequality constraints: l <= Ay <= u.
    ub_vec : numpy.ndarray
        The u vector of the linear inequality constraints: l <= Ay <= u.

    """
    a_eq_mat, b_eq_mat, b_eq_vec = constraints.equality_mat_vec(None)
    a_ineq_vec, b_ineq_mat, b_ineq_vec = constraints.inequality_mat_vec(None)

    a_mat = sparse.bmat([[a_eq_mat, b_eq_mat], [a_ineq_vec, b_ineq_mat]])
    lb_vec = np.concatenate([b_eq_vec, np.full_like(b_ineq_vec, -np.inf)])
    ub_vec = np.concatenate([b_eq_vec, b_ineq_vec])

    return a_mat.tocsc(), lb_vec, ub_vec
