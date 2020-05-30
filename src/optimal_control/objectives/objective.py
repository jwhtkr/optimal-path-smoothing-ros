"""Contains the base class for representing a cost function."""


# TODO: Refactor so that the instantaneous and terminal funcs are in the base class


class Objective(object):
    """
    Represent an objective function to be optimized.

    Can be a cost function (to be minimized) or a reward function (to be
    maximized). It is preferred to inherit from a child class designed as either
    one for the sake of being explicit.

    """
    pass


class ContinuousObjective(Objective):
    """
    Represent an objective function for continuous time optimization.

    Can be a cost function (to be minimized) or a reward function (to be
    maximized). It is preferred to inherit from a child class designed as either
    one for the sake of being explicit.

    """

    # def derivative(self, t_initial, t_final, state, ctrl):
    #     """
    #     Calculate the derivative of the objective w.r.t. the state and ctrl.

    #     Calculate the derivative of the objective with respect to both the state
    #     and the control input. Must be overidden by child classes. The `state`
    #     and `ctrl` arguments are functions of time that should be integrable in
    #     general.

    #     Parameters
    #     ----------
    #     t_initial : double
    #         The initial time for evaluating the derivative.
    #     t_final : double
    #         The final time for evaluating the derivative
    #     state : func
    #         The state as a function of time for evaluating the derivative.
    #     ctrl : func
    #         The control input as a function of time for evaluating the
    #         derivative.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         The derivative with respect to the state is returned as a numpy
    #         array of shape (1,len(state)).
    #     numpy.ndarray
    #         The derivative with respect to the control input is returned as a
    #         numpy array of shape (1,len(ctrl)).

    #     """
    #     raise NotImplementedError

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


class DiscreteObjective(Objective):
    """
    Represent an objective function for discrete time optimization.

    Can be a cost function (to be minimized) or a reward function (to be
    maximized). It is preferred to inherit from a child class designed as either
    one for the sake of being explicit.

    """

    # def derivative(self, k, state, ctrl):
    #     """
    #     Calculate the derivative of the objective w.r.t. the state and ctrl.

    #     Calculate the derivative of the objective with respect to both the state
    #     and the control input. Must be overidden by child classes.

    #     Parameters
    #     ----------
    #     k : int
    #         The time index at which to evalutate the derivative.
    #     state : numpy.ndarray
    #         The state at time `t` for which the derivative should be evaluated.
    #     ctrl : numpy.ndarray
    #         The control input at time `t` for which the derivative should be
    #         evalutated.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         The derivative with respect to the state is returned as a numpy
    #         array of shape (1,len(state)).
    #     numpy.ndarray
    #         The derivative with respect to the control input is returned as a
    #         numpy array of shape (1,len(ctrl)).

    #     """
    #     raise NotImplementedError

    def instantaneous(self, k, state, ctrl):
        """
        Calculate the instantaneous value of the objective function at time `t`.

        This must be overidden by child classes.

        Parameters
        ----------
        k : int
            The time index at which to evalutate the constraint.
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

    def terminal(self, k, state, ctrl):
        """
        Calculate the terminal value of the objective function at index `k`.

        This must be overidden by child classes.

        Parameters
        ----------
        k : int
            The time index at which to evalutate the constraint.
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
