"""Solver class that wraps the Gurobi optimization for Optimal Control use."""


import numpy as np

import gurobipy as gp

from optimal_control.solvers import solver_utils
from optimal_control.solvers import solver
from optimal_control.objectives import quadratic_cost
from optimal_control.constraints import linear_constraints
from optimal_control.constraints import binary_constraints
from optimal_control.constraints import constraints as constrs


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

    def __init__(self): # noqa: D107
        super(Gurobi, self).__init__()
        self.model = gp.Model()
        self.x_vec = None
        self.u_vec = None
        self.bin_vec = None
        self.obj = None
        # self.quadratic = None
        # self.linear = None
        # self.constraint_matrix = None
        # self.lower_bound = None
        # self.upper_bound = None
        self.is_setup = False

    def setup(self, objective, constraints, **kwargs):
        """
        Set up (or update) the optimization problem to solve.

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
        constraints : constraints.Constraints
            The constraints object to be converted to the Gurobi formulation
        **kwargs
            The keyword arguments to be passed to the internal Gurobi solver
            instance as settings. See the Gurobi documentation for more
            information
            https://www.gurobi.com/documentation/9.0/refman/py_python_api_overview.html.

        """
        if not isinstance(objective, quadratic_cost.QuadraticCost):
            raise TypeError("Gurobi can only solve optimization problems with a"
                            " quadratic cost for now.")
        # if not isinstance(constraints, linear_constraints.LinearConstraints):
        #     raise TypeError("Gurobi can only solve optimization problems with "
        #                     "linear constraints.")

        if not isinstance(objective,
                          quadratic_cost.DiscreteCondensedQuadraticCost):
            # if cost isn't condensed already, convert to the condensed form.
            objective = solver_utils.to_condensed(objective)

        # Convert to or extract the elements needed for the Gurobi formulation.

        # P, q = solver_utils.to_p_q(objective)
        # A, lb, ub = solver_utils.to_a_l_u(constraints)

        if not self.is_setup:
            self._setup(objective, constraints, **kwargs)

            self.is_setup = True
        else:
            # TODO: Adjust to only update what has changed, not the whole problem
            # self.solver.update(P, q, A, l, u)
            # self.solver.update_settings(**kwargs)
            raise RuntimeError("Updating Gurobi is not supported currently.")

    def _setup(self, objective, constraints, **kwargs):  # pylint: disable=unused-argument
        self.x_vec = self.model.addMVar(shape=int(constraints.n_state),
                                        vtype=gp.GRB.CONTINUOUS,
                                        lb=-np.inf,
                                        name="x")
        self.u_vec = self.model.addMVar(shape=int(constraints.n_ctrl),
                                        vtype=gp.GRB.CONTINUOUS,
                                        lb=-np.inf,
                                        name="u")

        self._make_objective(objective)
        self._make_constraints(constraints)

        # a_eq_mat, b_eq_mat, b_eq_vec = constraints.equality_mat_vec(None)
        # a_ineq_mat, b_ineq_mat, b_ineq_vec = constraints.inequality_mat_vec(None)
        # self.model.addConstr(a_eq_mat @ self.x_vec + b_eq_mat @ self.u_vec
        #                         == b_eq_vec.flatten(),
        #                         name="eq_constr")
        # self.model.addConstr(a_ineq_mat @ self.x_vec + b_ineq_mat @ self.u_vec
        #                         <= b_ineq_vec.flatten(),
        #                         name="ineq_constr")

    def _make_objective(self, objective):
        self.obj = objective.instantaneous(None, self.x_vec, self.u_vec)
        self.obj += objective.terminal(None, self.x_vec, self.u_vec)
        if isinstance(objective, quadratic_cost.cost.Cost):
            self.model.setObjective(self.obj, gp.GRB.MINIMIZE)
        else:
            self.model.setObjective(self.obj, gp.GRB.MAXIMIZE)

    def _make_constraints(self, constraints):
        has_binary = False
        for eq_constr in constraints.eq_constraints:
            if isinstance(eq_constr,
                          binary_constraints.BinaryEqualityConstraint):
                if not has_binary:
                    self.bin_vec = self.model.addMVar(shape=(int(eq_constr.n_binary),),
                                                      vtype=gp.GRB.BINARY,
                                                      name="binary")
                    has_binary = True

                constr_val = eq_constr.constraint(None, self.x_vec, self.u_vec,
                                                  self.bin_vec)
            else:
                constr_val = eq_constr.constraint(None, self.x_vec, self.u_vec)

            self.model.addConstr(constr_val == eq_constr.eq_val(None).flatten())

        for ineq_constr in constraints.ineq_constraints:
            if isinstance(ineq_constr,
                          binary_constraints.BinaryInequalityConstraint):
                if not has_binary:
                    self.bin_vec = self.model.addMVar(shape=(int(ineq_constr.n_binary),),
                                                      vtype=gp.GRB.BINARY,
                                                      name="binary")
                    has_binary = True

                constr_val = ineq_constr.constraint(None, self.x_vec,
                                                    self.u_vec, self.bin_vec)
            else:
                constr_val = ineq_constr.constraint(None, self.x_vec,
                                                    self.u_vec)

            self.model.addConstr(constr_val <= ineq_constr.bound(None).flatten())

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
            for key, val in kwargs.items():
                self.model.setParam(key, val)
        if warm_start:
            self._warm_start(warm_start)
        self.model.optimize()
        if self.model.getAttr("status") == gp.GRB.OPTIMAL:
            return np.concatenate([self.x_vec.x, self.u_vec.x])
        else:
            return None

    def _warm_start(self, warm_start_vec):
        """Warm start the optimization model."""
        del warm_start_vec
        print("Warm start is currently not available. Solving without warm "
              + "start.")
