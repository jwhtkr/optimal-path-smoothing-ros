"""Contains the base class for an optimization solver wrapper class."""



class Solver(object):
    """
    Represent an optimization solver in an abstract way.

    Child classes will wrap an optimization class or library and provide the
    interface defined in this base class. Because the solver is instantiated on
    its own (before the problem is composed/problem info is available) it is
    required to call the `setup` method before calling the `solve` method. This
    can also be used in cases when the problem needs to be updated with
    modified parameters (MPC).

    Attributes
    ----------
    solver : obj
        The optimization solver for use in solving the optimal control problem.

    """
    def __init__(self):
        self.solver = None

    def setup(self, objective, constraints, **kwargs):
        """
        Perform any necessary setup for the optimization solver.

        Must be overidden by child classes. The function signature is flexible
        for any method of passing parameters to the child class implementation.

        Parameters
        ----------
        objective : optimal_control.objectives.objective.Objective
            An objective to be optimized. Converted in this method to a form
            compatible with the internal solver.
        constraints : optimal_control.constraints.constraints.Constraints
            A set of constraints that must be satisfied. Converted in this
            method to a form compatible with the internal solver.
        **kwargs
            Keyword arguments for use by the child class for setup. Some may be
            passed to a setup function of the internal optimization solver.

        Returns
        -------
        any or None
            Returns what is returned by the internal optimization solver setup
            method if it has one. Otherwise returns None.
        """
        raise NotImplementedError

    def solve(self, **kwargs):
        """
        Solve the optimal control optimization problem.

        Must be overidden by child classes. The function signature is flexible
        for any method of passing parameters to the child class implementation
        or internal solver class.

        Parameters
        ----------
        **kwargs
            Keyword arguments for use by the child class for solving. Some may
            be passed to the solve function of the internal optimization solver.

        Returns
        -------
        numpy.ndarray or None
            Returns the solution to the optimal control optimization problem, or
            None if no solution was found (infeasible).
        """
        raise NotImplementedError
