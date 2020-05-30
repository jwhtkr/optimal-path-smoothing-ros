"""Contains classes for performing direct optimal control methods."""


from optimal_control.opt_ctrl import optimal_control_base as opt_ctrl_base


class DirectOptimalControl(opt_ctrl_base.OptimalControlBase):
    """
    Represent an optimal control problem to be solved with direct methods.

    The direct method of optimal control can be thought of as
    discretize-then-optimize. Thus, this class handles necessary routines
    related to first discretizing the problem representation, formulating
    the correct optimization problem, and then solving the problem with the
    given solver.

    Attributes
    ----------
    dynamics : Dynamics
        A subclass of the Dynamics class that represents the dynamics of the
        system.
    constraints : Constraints
        An instance or subclass of Constraints that represents the constraints
        of the optimization problem.
    cost : Cost
        An instance or subclass of the Cost class to represent the cost or
        objective to be optimized.
    solver : Solver
        A subclass of the Solver class that wraps an optimization solver for use
        in optimal control.
    kwargs : dict of {str: any}
        A dictionary of key-word arguments to be passed to the underlying solver
        at solve time.
    direct_constraints : Constraints
        An instance or subclass of Constraints that represents the constraints
        of the direct optimization problem. Transformed/discretized from
        :attribute:`constraints`, and includes the dynamic constraints
    direct_cost : Cost
        An instance or subclass of the Cost class that represents the cost
        function of the direct optimal control problem. Transformed and
        discretized from :attribute:`cost`.

    Parameters
    ----------
    dynamics : Dynamics
        A subclass of the Dynamics class that represents the dynamics of the
        system.
    constraints : Constraints
        An instance or subclass of Constraints that represents the constraints
        of the optimization problem.
    cost : Cost
        An instance or subclass of the Cost class to represent the cost or
        objective to be optimized.
    solver : Solver
        A subclass of the Solver class that wraps an optimization solver for use
        in optimal control.
    **kwargs
        The keyword arguments to be passed to the underlying solver at solve
        time.

    """
    def __init__(self, dynamics, constraints, cost, solver, **kwargs):
        self.direct_cost = None
        self.direct_constraints = None
        super(DirectOptimalControl, self).__init__(dynamics, constraints, cost,
                                                   solver, **kwargs)
        self.update()
        self.solver.setup(objective=self.direct_cost,
                          constraints=self.direct_constraints)

    def update(self):
        """
        Update the problem representation with the current attributes.

        This can be a fairly intensive process depending on the problem size, so
        only call when necessary (i.e., the problem has actually changed with
        new dynamics/cost/etc.). Results in storing :attribute:`direct_cost` and
        :attribute:`direct_constraints`.

        """
        raise NotImplementedError

    def solve(self, warm_start=None):
        """See base class."""
        return self.solver.solve(warm_start=warm_start, **self.kwargs)


# TODO: Make simultaneous and Sequential children of this
