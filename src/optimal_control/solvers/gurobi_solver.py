"""Solver class that wraps the Gurobi optimization for Optimal Control use."""


import numpy as np

import gurobipy as gp

import optimal_control.sparse_utils as sparse
from optimal_control.solvers import solver
from optimal_control.objectives import quadratic_cost
from optimal_control.constraints import linear_constraints


class Gurobi(solver.Solver):
    """
    Wrap the Gurobi solver for use in Optimal Control.

    The Gurobi solver is a Linear or Quadratic Program Mixed Integer solver,
    and so is inherently limited to control problems that have a linear or
    quadratic objective, linear dynamics, and linear or quadratic constraints,
    but allows for using binary or integer variables in the formulation. This is
    a somewhat restrictive set of problems, but can often be used as a quick
    approximation or locally applicable solution.

    Attributes
    ----------
    model : gurobipy.model
        The internal instance of the Gurobi model to be optimized.
    quadratic : numpy.ndarray or scipy.sparse.spmatrix
        The P matrix of the cost J(y) = y^T P y + q^T y.
    linear : numpy.ndarray
        The q vector of the cost J(y) = y^T P y + q^T y.
    constraint_matrix : numpy.ndarray or scipy.sparse.spmatrix
        The A matrix of the non-integer constraints l <= A y <= u.
    lower_bound : numpy.ndarray
        The l vector of the non-integer constraints l <= A y <= u.
    upper_bound : numpy.ndarray
        The u vector of the non-integer constraints l <= A y <= u.
    is_setup : bool
        Indicates if setup has been called or not.

    """
    def __init__(self):
        super(Gurobi, self).__init__()
        self.model = gp.Model()
        # self.quadratic = None
        # self.linear = None
        # self.constraint_matrix = None
        # self.lower_bound = None
        # self.upper_bound = None
        self.is_setup = False

    def setup(self, objective, constraints, **kwargs):
        """
        Setup (or update) the optimization problem to solve.

        The arguments `objective` and `constraints` are used to convert to the
        objective and constraints of the gurobi solver and setup the internal
        solver/model instance. The objective must be a quadratic (or linear)
        cost, and the constraints must be quadratic, linear, integer, or one of
        Gurobi's special function or general constraints. Any keyword arguments
        are passed as settings to the internal solver, see Gurobi documentation
        for more information
        https://www.gurobi.com/documentation/9.0/refman/py_python_api_overview.html.

        Parameters
        ----------
        objective : quadratic_cost.DiscreteQuadraticCost
            The cost object to be converted to the Gurobi formulation.
        constraints : linear_constraints.LinearConstraints
            The constraints object to be converted to the Gurobi formulation
        **kwargs
            The keyword arguments to be passed to the internal Gurobi solver
            instance as settings. See the Gurobi documentation for more
            information
            https://www.gurobi.com/documentation/9.0/refman/py_python_api_overview.html.

        """
        if not isinstance(objective, quadratic_cost.QuadraticCost):
            raise TypeError("Gurobi can only solve optimization problems with a "
                            "quadratic cost.")
        if not isinstance(constraints, linear_constraints.LinearConstraints):
            raise TypeError("Gurobi can only solve optimization problems with "
                            "linear constraints.")

        # if not isinstance(objective,
        #                   quadratic_cost.DiscreteCondensedQuadraticCost):
        #     # if cost isn't condensed already, convert to the condensed form.
        #     objective = _to_condensed(objective)

        # Convert to or extract the elements needed for the Gurobi formulation.
        P, q = _to_p_q(objective)
        A, l, u = _to_a_l_u(constraints)

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
            Gurobi solver. For more information on the available settings see
            the Gurobi documentation
            https://www.gurobi.com/documentation/9.0/refman/py_python_api_overview.html.
            Additionally, a warm-start solution can be provided in the kwargs
            with the key "warm_start".

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
