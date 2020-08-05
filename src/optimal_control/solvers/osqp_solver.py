"""Wrap the OSQP solver for use with Optimal Control."""


import optimal_control.sparse_utils as sparse
import numpy as np

import osqp

from optimal_control.solvers import solver
from optimal_control.objectives import quadratic_cost
from optimal_control.constraints import linear_constraints


def _to_condensed(cost):
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


def _to_p_q(cost):
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
        The P matrix of a quadratic cost of form J(y) = y^T P y + q^T y.
    q_mat : numpy.ndarray
        The q vector of a quadratic cost of form J(y) = y^T P y + q^T y.

    """
    q_mat = sparse.coo_matrix(cost.inst_state_cost)
    r_mat = sparse.coo_matrix(cost.inst_ctrl_cost)
    s_mat = sparse.coo_matrix(cost.term_state_cost)
    qs_mat = q_mat + s_mat
    x_d = cost.desired_state(None)
    u_d = cost.desired_ctrl(None)

    p_mat = sparse.bmat([[qs_mat, None], [None, r_mat]])
    q_mat = np.concatenate([-qs_mat.transpose().dot(x_d),
                        -r_mat.transpose().dot(u_d)])

    return p_mat.tocsc(), q_mat


def _to_a_l_u(constraints):
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
    l_mat : numpy.ndarray
        The l vector of the linear inequality constraints: l <= Ay <= u.
    u_mat : numpy.ndarray
        The u vector of the linear inequality constraints: l <= Ay <= u.

    """
    a_eq, b_eq, _b_eq = constraints.equality_mat_vec(None)
    a_ineq, b_ineq, _b_ineq = constraints.inequality_mat_vec(None)

    a_mat = sparse.bmat([[a_eq, b_eq], [a_ineq, b_ineq]])
    l_mat = np.concatenate([_b_eq, np.full_like(_b_ineq, -np.inf)])
    u_mat = np.concatenate([_b_eq, _b_ineq])

    return a_mat.tocsc(), l_mat, u_mat


class OSQP(solver.Solver):
    """
    Wrap the OSQP solver for use in Optimal Control.

    The OSQP solver is a Quadratic Program solver, and so is inherently limited
    to control problems that have a linear or quadratic objective, linear
    dynamics, and linear constraints. This is a restrictive set of problems, but
    can often be used as a quick approximation or locally applicable solution.
    Because the solver is instantiated on its own (before the problem is
    composed/problem info is available) it is required to call the `setup`
    method before calling the `solve` method. This can also be used in cases
    when the problem needs to be updated (MPC) with modified parameters.

    Attributes
    ----------
    solver : osqp.OSQP
        The internal instance of the OSQP solver.
    quadratic : numpy.ndarray or scipy.sparse.spmatrix
        The P matrix of the OSQP cost J(y) = y^T P y + q^T y.
    linear : numpy.ndarray
        The q vector of the OSQP cost J(y) = y^T P y + q^T y.
    A : numpy.ndarray or scipy.sparse.spmatrix
        The A matrix of the OSQP constraints l <= A y <= u.
    lower_bound : numpy.ndarray
        The l vector of the OSQP constraints l <= A y <= u.
    upper_bound : numpy.ndarray
        The u vector of the OSQP constraints l <= A y <= u.
    is_setup : bool
        Indicates if setup has been called or not.
    kwargs : dict of {str: any}
        Keyword arguments to be passed to the set-up

    Parameters
    ----------
    **kwargs
        Keyword arguments to be passed to the internal solver instance at set-up
        time.

    """

    def __init__(self): # noqa: D107
        super(OSQP, self).__init__()
        self.solver = osqp.OSQP()
        self.quadratic = None
        self.linear = None
        self.constraint_matrix = None
        self.lower_bound = None
        self.upper_bound = None
        self.is_setup = False

    def setup(self, objective, constraints, **kwargs):
        """
        Create or update the optimization problem.

        The arguments `objective` and `constraints` are used to convert to the
        parameters of the OSQP solver, P, q, A, l, u and setup the internal
        solver instance. The objective must be a quadratic cost, and the
        constraints must be linear. Any keyword arguments are passed as settings
        to the internal solver, see OSQP documentation for more information
        (https://osqp.org/docs).

        Parameters
        ----------
        objective : quadratic_cost.DiscreteQuadraticCost
            The cost object to be converted to the OSQP formulation.
        constraints : linear_constraints.LinearConstraints
            The constraints object to be converted to the OSQP formulation
        **kwargs
            The keyword arguments to be passed to the internal OSQP solver
            instance as settings. See the OSQP documentation
            (https://osqp.org/docs) for more information.

        """
        if not isinstance(objective, quadratic_cost.QuadraticCost):
            raise TypeError("OSQP can only solve optimization problems with a "
                            "quadratic cost.")
        if not isinstance(constraints, linear_constraints.LinearConstraints):
            raise TypeError("OSQP can only solve optimization problems with "
                            "linear constraints.")

        if not isinstance(objective,
                          quadratic_cost.DiscreteCondensedQuadraticCost):
            # if cost isn't condensed already, convert to the condensed form.
            objective = _to_condensed(objective)

        # Convert to/extract the elements needed for the OSQP formulation.
        p_mat, q_mat = _to_p_q(objective)
        a_mat, l_mat, u_mat = _to_a_l_u(constraints)

        if not self.is_setup:
            self.solver.setup(p_mat, q_mat, a_mat, l_mat, u_mat, **kwargs)
            self.is_setup = True
        else:
            # TODO: Adjust to only update what has changed, not the whole problem
            self.solver.update(p_mat, q_mat, a_mat, l_mat, u_mat)
            self.solver.update_settings(**kwargs)

        self.quadratic = p_mat
        self.linear = q_mat
        self.constraint_matrix = a_mat
        self.lower_bound = l_mat
        self.upper_bound = u_mat

    def solve(self, **kwargs):
        """
        Solve the optimization problem.

        Parameters
        ----------
        **kwargs
            If this is not empty, it is used to update the settings of the
            OSQP solver. For more information on the available settings see the
            OSQP documentation (https://osqp.org/docs).
        Returns
        -------
        numpy.ndarray or None
            the solution to the optimal control optimization problem, or
            None if no solution was found (infeasible).

        """
        warm_start = kwargs.pop("warm_start")
        if kwargs:
            self.solver.update_settings(**kwargs)
        if warm_start:
            self.solver.warm_start(x=warm_start)
        results = self.solver.solve()
        return results.x
