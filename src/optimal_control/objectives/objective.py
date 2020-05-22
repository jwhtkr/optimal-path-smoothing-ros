"""Contains the base class for representing a cost function."""



class Objective(object):
    """
    Represent an objective function to be optimized.

    Can be a cost function (to be minimized) or a reward function (to be
    maximized). It is preferred to inherit from a child class designed as either
    one for the sake of being explicit.

    Attributes
    ----------
    params : dict of {str: any}
        A dictionary of parameters used in calculating the cost and its
        derivatives.

    Parameters
    ----------
    params : dict of {str: any}
        A dictionary of parameters used in calculating the cost and its
        derivatives.

    """
    def __init__(self, params):
        self.params = params

    def instantaneous(self, t, state, ctrl):
        """
        Calculate the instantaneous value of the objective function at time `t`.

        This must be overidden by child classes.

        Parameters
        ----------
        t : double
            The time at which to evalutate the constraint.
        state : numpy.ndarray
            The state at time `t` for which the constraint should be evaluated.
        ctrl : numpy.ndarray
            The control input at time `t` for which the constraint should be
            evalutated.

        Returns
        -------
        double
            The instantaneous value of the objective function at time `t`.

        """
        raise NotImplementedError

    def terminal(self, t, state, ctrl):
        """
        Calculate the terminal value of the objective function at time `t`.

        This must be overidden by child classes.

        Parameters
        ----------
        t : double
            The time at which to evalutate the constraint.
        state : numpy.ndarray
            The state at time `t` for which the constraint should be evaluated.
        ctrl : numpy.ndarray
            The control input at time `t` for which the constraint should be
            evalutated.

        Returns
        -------
        double
            The value of the terminal portion of the objective function at time
            `t`.

        """
        raise NotImplementedError

    def derivative(self, t, state, ctrl):
        """
        Calculate the derivative of the objective w.r.t. the state and ctrl.

        Calculate the derivative of the objective with respect to both the state
        and the control input. Must be overidden by child classes.

        Parameters
        ----------
        t : double
            The time at which to evalutate the constraint.
        state : numpy.ndarray
            The state at time `t` for which the constraint should be evaluated.
        ctrl : numpy.ndarray
            The control input at time `t` for which the constraint should be
            evalutated.

        Returns
        -------
        numpy.ndarray
            The derivative with respect to the state is returned as a numpy
            array of shape (1,len(state)).
        numpy.ndarray
            The derivative with respect to the control input is returned as a
            numpy array of shape (1,len(ctrl)).

        """
        raise NotImplementedError
