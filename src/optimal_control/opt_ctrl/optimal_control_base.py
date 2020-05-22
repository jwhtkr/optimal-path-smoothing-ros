"""Contains the base object for an optimal control problem."""


class OptimalControlBase(object):
    """
    A base class for representing optimal control problems.

    This base class represents an optimal control problem from a high level. The
    class makes no assumptions about method, solver, or type of problem, and
    *should* be able to function if the passed in parameters have the necessary
    capabilities. However, it is meant to be subclassed to provide more specific
    scenario and problem capabilities and functionality.

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
    kwargs : dict
        A dictionary of key-word arguments to be passed to the underlying solver
        at solve time.


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

    """
    def __init__(self, dynamics, constraints, cost, solver, **kwargs):
        self.dynamics = dynamics
        self.constraints = constraints
        self.cost = cost
        self.solver = solver
        self.kwargs = kwargs

        self.update()

    def update(self):
        """
        Update problem representation based on the current attributes.

        Update internal representations of the problem parameters according to
        the current attributes. This method should be called after changing any
        of the attributes to ensure a solution is valid for the current state.
        """
        pass

    def solve(self, warm_start=None):
        """
        Solve the optimal control problem. Return the optimal problem solution.

        Use the solver class to solve the optimal control problem with the
        stored key-word arguments. A warm start control sequence can be passed
        in to improve solve time.

        Parameters
        ----------
        warm_start : numpy.ndarray
            An initial guess at the optimal solution.

        Returns
        -------
        numpy.ndarray
            The solution to the optimal control problem.
        """
        pass
