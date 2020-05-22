"""Contains the base class for an optimization solver wrapper class."""



class SolverBase(object):
    """
    Represent an optimization solver in an abstract way.

    Child classes will wrap an optimization class or library and provide the
    interface defined in this base class.

    Attributes
    ----------
    solver : obj
        The optimization solver for use in solving the optimal control problem.

    """
    def __init__(self):
        self.solver = None

    def setup(self, *args, **kwargs):
        """
        Perform any necessary setup for the optimization solver.

        Must be overidden by child classes, and the function signature is
        flexible for any method of passing parameters to the child class
        implementation.

        Parameters
        ----------
        *args
            Positional arguments for use by the child class for setup. Some may
            be passed to a setup function of the internal optimization solver.
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
