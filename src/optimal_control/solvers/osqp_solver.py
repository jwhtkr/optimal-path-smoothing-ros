"""Wrap the OSQP solver for use with Optimal Control."""


import osqp

from optimal_control.solvers import solver_utils
from optimal_control.solvers import solver
from optimal_control.objectives import quadratic_cost
from optimal_control.constraints import linear_constraints


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
        The P matrix of the OSQP cost J(y) = y^T P y + 2 q^T y.
    linear : numpy.ndarray
        The q vector of the OSQP cost J(y) = y^T P y + 2 q^T y.
    A : numpy.ndarray or scipy.sparse.spmatrix
        The A matrix of the OSQP constraints l <= A y <= u.
    lower_bound : numpy.ndarray
        The l vector of the OSQP constraints l <= A y <= u.
    upper_bound : numpy.ndarray
        The u vector of the OSQP constraints l <= A y <= u.
    is_setup : bool
        Indicates if setup has been called or not.

    """
    def __init__(self):
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
        Setup (or update) the optimization problem to solve.

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
            objective = solver_utils.to_condensed(objective)

        # Convert to/extract the elements needed for the OSQP formulation.
        P, q = solver_utils.to_p_q(objective)
        A, l, u = solver_utils.to_a_l_u(constraints)

        if not self.is_setup:
            self.solver.setup(P, q, A, l, u, **kwargs)
            self.is_setup = True
        else:
            # TODO: Adjust to only update what has changed, not the whole problem
            self.solver.update(P, q, A, l, u)
            self.solver.update_settings(**kwargs)

        self.quadratic = P
        self.linear = q
        self.constraint_matrix = A
        self.lower_bound = l
        self.upper_bound = u

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
