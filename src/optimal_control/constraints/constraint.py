"""Contains the classes for representing individual optimization constraints."""

import functools
import numpy as np


def invert(func):
    """
    Invert the value returned by a function. To be used as a decorator.

    Parameters
    ----------
    func : function
        A function to be inverted.

    Returns
    -------
    function
        A function that returns the inverted value of the argument function.

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Invert the return of a function."""
        return -func(*args, **kwargs)
    return wrapper


class Constraint(object):
    """
    Represents a general constraint on state/control input for optimal control.

    Represents a general constraint on either state or control input. Can be
    time-varying and non-linear or linear. Can be either equality or inequality
    constrained. This is meant as a base class, with child classes being, e.g.,
    a linear inequality constraint, a non-linear equality constraint, etc.

    Attributes
    ----------
    n_state : int
        The number of states of the constraint.
    n_ctrl : int
        The number of control inputs in the constraint.

    Parameters
    ----------
    n_state : int
        The number of states of the constraint.
    n_ctrl : int
        The number of control inputs in the constraint.

    """

    def __init__(self, n_state, n_ctrl): # noqa: D107
        self.n_state = n_state
        self.n_ctrl = n_ctrl

    def is_satisfied(self, t, state, ctrl):
        """
        Return a boolean indicating if the constraint is satisfied or violated.

        The inputs are used to determine if the constraint is violated or not.
        The typical use will be to determine the output of the constraint
        function and then to compare to a value (often 0) for equality or
        inequality.

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
        bool
            Indicates if the constraint is satisfied (True) or violated (False).

        """
        raise NotImplementedError

    def constraint(self, t, state, ctrl):
        """
        Return the value(s) of the constraint function.

        Utilize the time, state, and input control variables to calculate the
        value of the constraint function. Can return a vector or scalar value.

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
        double or np.array of doubles
            The value of the evaluation of the constraint function.

        """
        raise NotImplementedError

    # def derivative(self, t, state, ctrl):
    #     """
    #     Return the derivative of the constraint w.r.t. state/control input.

    #     Calculate the derivate of the constraint with respect to both state and
    #     control input and return as two numpy arrays

    #     Parameters
    #     ----------
    #     t : double
    #         The time at which to evalutate the derivative of the constraint.
    #     state : numpy.ndarray
    #         The state at time `t` for which the derivative should be evaluated.
    #     ctrl : numpy.ndarray
    #         The control input at time `t` for which the derivative should be
    #         evalutated.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         The derivative with respect to the state as an array of shape
    #         (1, len(state))
    #     numpy.ndarray
    #         The derivative with respect to the control input as an array of
    #         shape (1, len(ctrl))

    #     """
    #     raise NotImplementedError


class EqualityConstraint(Constraint):  # pylint: disable=abstract-method
    """
    Represent an equality constraint.

    Represent an equality constraint on states and control inputs of the system.
    Can be non-linear or linear. Can be time-varying or time-invariant.

    Attributes
    ----------
    n_state : int
        The number of states of the constraint.
    n_ctrl : int
        The number of control inputs in the constraint.
    eq_val : func
        A function of time that returns the desired value of the constraint.
        I.e.: h(t, state, ctrl) = eq_val(t).
    eps : double
        Determines the allowable tolerance of the constraint, as in:
        abs(h(t, state, ctrl) - eq_val(t)) < eps. Its default value is 1e-6.

    Parameters
    ----------
    n_state : int
        The number of states of the constraint.
    n_ctrl : int
        The number of control inputs in the constraint.
    eq_val : func
        A function of time that returns the desired value of the constraint.
        I.e.: h(t, state, ctrl) = eq_val(t).
    eps : double, default 1e-6
        Determines the allowable tolerance of the constraint, as in:
        abs(h(t, state, ctrl) - eq_val(t)) < eps. Its default value is 1e-6.

    """

    def __init__(self, n_state, n_ctrl, eq_val, eps=1e-6): # noqa: D107
        super(EqualityConstraint, self).__init__(n_state, n_ctrl)
        self.eq_val = eq_val
        self.eps = eps

    def is_satisfied(self, t, state, ctrl):
        """See base class."""
        error = abs(self.constraint(t, state, ctrl) - self.eq_val(t))
        return np.all(error < self.eps)


class InequalityConstraint(Constraint):  # pylint: disable=abstract-method
    """
    Represents an inequality constraint.

    Represent an inequality constraint on states and control inputs of the
    system. Can be linear, non-linear, time-invariant, or time-varying. Only
    upper-bounded inequality constraints are supported at this time.
    Lower-bounded inequality constraints can be written as upper-bounded
    constraints by inverting both the constraint function and the bound. A
    convenient decorator `invert` is provided in the module containing this
    class. Thus a convenient way to create a lower-bounded subclass would be to
    use the `invert` decorator with the `constraint` method and also the `bound`
    function before being passed to the parent class constructor.

    Attributes
    ----------
    n_state : int
        The number of states of the constraint.
    n_ctrl : int
        The number of control inputs in the constraint.
    bound : func
        A function of time that returns the converted upper bound constraint
        used to make all inequality constraints consistently of the form
        g'(t, state, ctrl) <= bound. Lower bounded constraints are transformed
        as described above.

    Parameters
    ----------
    n_state : int
        The number of states of the constraint.
    n_ctrl : int
        The number of control inputs in the constraint.
    upper_bound : func
        A function of time that returns the upper bound of an inequality
        constraint. The constraint is assumed to be of the form
        g(t, state, ctrl) <= upper_bound.

    """

    def __init__(self, n_state, n_ctrl, upper_bound): # noqa: D107
        super(InequalityConstraint, self).__init__(n_state, n_ctrl)
        self.bound = upper_bound

    def is_satisfied(self, t, state, ctrl):
        """See base class."""
        return np.all(self.constraint(t, state, ctrl) <= self.bound(t))
