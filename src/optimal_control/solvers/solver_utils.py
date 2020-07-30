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
    P : scipy.sparse.csc_matrix
        The P matrix of a quadratic cost of form J(y) = y^T P y + 2 q^T y.
    q : numpy.ndarray
        The q vector of a quadratic cost of form J(y) = y^T P y + 2 q^T y.

    """
    Q = sparse.coo_matrix(cost.inst_state_cost)
    R = sparse.coo_matrix(cost.inst_ctrl_cost)
    S = sparse.coo_matrix(cost.term_state_cost)
    QS = Q + S
    x_d = cost.desired_state(None)
    u_d = cost.desired_ctrl(None)

    # P = sparse.bmat([[QS, None], [None, R]])
    P = sparse.block_diag([QS, R])
    q = np.concatenate([-QS.transpose().dot(x_d),
                        -R.transpose().dot(u_d)])

    return P.tocsc(), q


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
    A : scipy.sparse.csc_matrix
        The A matrix of the linear inequality constraints: l <= Ay <= u.
    l : numpy.ndarray
        The l vector of the linear inequality constraints: l <= Ay <= u.
    u : numpy.ndarray
        The u vector of the linear inequality constraints: l <= Ay <= u.

    """
    A_eq, B_eq, b_eq = constraints.equality_mat_vec(None)
    A_ineq, B_ineq, b_ineq = constraints.inequality_mat_vec(None)

    A = sparse.bmat([[A_eq, B_eq], [A_ineq, B_ineq]])
    l = np.concatenate([b_eq, np.full_like(b_ineq, -np.inf)])
    u = np.concatenate([b_eq, b_ineq])

    return A.tocsc(), l, u
