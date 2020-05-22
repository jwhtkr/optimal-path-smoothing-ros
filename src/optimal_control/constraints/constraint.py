"""Contains the classes for representing individual optimization constraints."""

import functools


def _invert(func):
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
        return - func(*args, **kwargs)
    return wrapper


class Constraint(object):
    """
    Represents a general constraint on state/control input for optimal control.

    Represents a general constraint on either state or control input. Can be
    time- varying and non-linear or linear. Can be either equality or inequality
    constrained. This is meant as a base class, with child classes being, e.g.,
    a linear inequality constraint, a non-linear equality constraint, etc.

    Attributes
    ----------
    params : dict of {str: any}
        A dictionary of parameters for use in the constraint equation.

    Parameters
    ----------
    params : dict of {str: any}, optional
        A dictionary of parameters for use in the constraint equation.

    """
    def __init__(self, params=None):
        self.params = params

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
        Return the value of the constraint function.

        Utilize the input variables and the stored kwarg attribute to calculate
        the value of the constraint function.

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

    def derivative(self, t, state, ctrl):
        """
        Return the derivative of the constraint w.r.t. state/control input.

        Calculate the derivate of the constraint with respect to both state and
        control input and return as two numpy arrays

        Parameters
        ----------
        t : double
            The time at which to evalutate the derivative of the constraint.
        state : numpy.ndarray
            The state at time `t` for which the derivative should be evaluated.
        ctrl : numpy.ndarray
            The control input at time `t` for which the derivative should be
            evalutated.

        Returns
        -------
        numpy.ndarray
            The derivative with respect to the state as an array of shape
            (1, len(state))
        numpy.ndarray
            The derivative with respect to the control input as an array of
            shape (1, len(ctrl))

        """
        raise NotImplementedError


class EqualityConstraint(Constraint):  # pylint: disable=abstract-method
    """
    Represent an equality constraint.

    Represent an equality constraint on states and control inputs of the system.
    Can be non-linear or linear. Can be time-varying or time-invariant.

    Attributes
    ----------
    eq_val : double
        Represents the desired value of the constraint. I.e.:
        h(t, state, ctrl) = eq_val.
    eps : double
        Determines the allowable tolerance of the constraint, as in:
        abs(h(t, state, ctrl) - eq_val) < eps. Its default value is 1e-6.
    params : dict of {str: any}
        A dictionary of parameters for use in the constraint equation.

    Parameters
    ----------
    eq_val : double
        Represents the desired value of the constraint. I.e.:
        h(t, state, ctrl) = eq_val.
    eps : double, default 1e-6
        Determines the allowable tolerance of the constraint, as in:
        abs(h(t, state, ctrl) - eq_val) < eps. Its default value is 1e-6.
    params : dict of {str: any}, optional
        A dictionary of parameters for use in the constraint equation.
    """
    def __init__(self, eq_val, eps=1e-6, params=None):
        super(EqualityConstraint, self).__init__(params)
        self.eq_val = eq_val
        self.eps = eps

    def is_satisfied(self, t, state, ctrl):
        """See base class."""
        return abs(self.constraint(t, state, ctrl) - self.eq_val) < self.eps


class InequalityConstraint(Constraint):  # pylint: disable=abstract-method
    """
    Represents an inequality constraint.

    Represent an inequality constraint on states and control inputs of the
    system. Can be linear, non-linear, time-invariant, or time-varying. One of
    upper_bound or lower_bound must be specified, but not both. Internally,
    a lower bounded constraint (i.e., lower_bound <= g(t,x,u)) is transformed to
    an upper bounded constraint by -g(t, state, ctrl) <= -lower_bound. Child
    classes, however, should not write the overides of :method:`constraint` and
    :method:`derivative` any differently than if they were being written without
    this knowledge, the conversion is automatic and part of the
    :class:`InequalityConstraint` definition.

    Attributes
    ----------
    upper_bound : double
        The upper bound of an inequality constraint. The constraint is assumed
        to be of the form g(t, state, ctrl) <= upper_bound. If upper_bound is
        not specified, then lower_bound must be specified.
    lower_bound : double
        The lower bound of an inequality constraint. The constraint is assumed
        t o be of the form g(t, state, ctrl) >= lower_bound. If lower_bound is
        not specified, then upper_bound must be specified.
    params : dict of {str: any}
        A dictionary of parameters for use in the constraint equation.
    bound : double
        The converted upper bound constraint used to make all inequality
        constraints consistently of the form g'(t, state, ctrl) <= bound. Lower
        bounded constraints are transformed as described above.

    Parameters
    ----------
    upper_bound : double, optional
        The upper bound of an inequality constraint. The constraint is assumed
        to be of the form g(t, state, ctrl) <= upper_bound. If upper_bound is
        not specified, then lower_bound must be specified.
    lower_bound : double, optional
        The lower bound of an inequality constraint. The constraint is assumed
        t o be of the form g(t, state, ctrl) >= lower_bound. If lower_bound is
        not specified, then upper_bound must be specified.
    params : dict of {str: any}, optional
        A dictionary of parameters for use in the constraint equation.

    """
    def __init__(self, upper_bound=None, lower_bound=None, params=None):
        if ((not upper_bound and not lower_bound)
                or (upper_bound and lower_bound)):
            # if both or neither upper_bound and lower_bound are specified
            # raise an error
            raise TypeError("One and only one of upper_bound and lower_bound "
                            "must be specified.")
        super(InequalityConstraint, self).__init__(params=params)
        if upper_bound:
            self.bound = upper_bound
        else:
            self.bound = -lower_bound  # pylint: disable=invalid-unary-operand-type
            self.constraint = _invert(self.constraint)
            self.derivative = _invert(self.derivative)

    def is_satisfied(self, t, state, ctrl):
        """See base class."""
        return self.constraint(t, state, ctrl) <= self.bound
