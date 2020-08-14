"""Represent constraints involving binary variables."""


import numpy as np

from optimal_control.constraints import constraint


class BinaryEqualityConstraint(constraint.EqualityConstraint):
    """
    Represent an equality constraint with binary variables.

    Attributes
    ----------
    See base class.
    n_binary: int
        The number of binary variables of the constraint.

    Parameters
    ----------
    See base class.
    n_binary: int
        The number of binary variables of the constraint.
    """

    def __init__(self, n_binary, n_state, n_ctrl, eq_val, eps=1e-6):  # noqa: D107
        super().__init__(n_state, n_ctrl, eq_val, eps)
        self.n_binary = n_binary

    def is_satisfied(self, t, state, ctrl, binary):  # pylint: disable=arguments-differ
        """
        Determine if the constraint is satisfied.

        Functionally the same as in the base class, with the additional function
        parameter of the binary variables to use.

        Parameters
        ----------
        See base class.
        binary : numpy.ndarray of bool
            The binary variables to evaluate the constraint satisfaction at.

        Returns
        -------
        See base class.
        """
        error = abs(self.constraint(t, state, ctrl, binary) - self.eq_val(t))
        return np.all(error < self.eps)

    def constraint(self, t, state, ctrl, binary):  # pylint: disable=arguments-differ
        """
        Return the value of the constraint function at time `t`.

        Functionally the same as the base class, but adds the binary variables
        to the function signature.

        Parameters
        ----------
        See base class.
        binary : numpy.ndarray or other array-like
            A vector of binary variables as input to the constraint function.

        Returns
        -------
        See base class

        """
        raise NotImplementedError


class BinaryInequalityConstraint(constraint.InequalityConstraint):
    """
    Represent an inequality constraint with binary variables.

    Attributes
    ----------
    See base class.
    n_binary: int
        The number of binary variables of the constraint.

    Parameters
    ----------
    See base class.
    n_binary: int
        The number of binary variables of the constraint.
    """

    def __init__(self, n_binary, n_state, n_ctrl, bound):  # noqa: D107
        super().__init__(n_state, n_ctrl, bound)
        self.n_binary = n_binary

    def is_satisfied(self, t, state, ctrl, binary):  # pylint: disable=arguments-differ
        """
        Determine if the constraint is satisfied.

        Functionally the same as in the base class, with the additional function
        parameter of the binary variables to use.

        Parameters
        ----------
        See base class.
        binary : numpy.ndarray of bool
            The binary variables to evaluate the constraint satisfaction at.

        Returns
        -------
        See base class.
        """
        return np.all(self.constraint(t, state, ctrl, binary) <= self.bound(t))

    def constraint(self, t, state, ctrl, binary):  # pylint: disable=arguments-differ
        """
        Return the value of the constraint function at time `t`.

        Functionally the same as the base class, but adds the binary variables
        to the function signature.

        Parameters
        ----------
        See base class.
        binary : numpy.ndarray or other array-like
            A vector of binary variables as input to the constraint function.

        Returns
        -------
        See base class

        """
        raise NotImplementedError


class ImplicationInequalityConstraint(BinaryInequalityConstraint):
    """
    Represent a constraint where a binary variable implies an inequality.

    A single binary variable is used to determine whether or not an inequality
    constraint must hold. Big-M formulations can then be used to represent this
    implication with an inequality constraint including the binary variable.

    Attributes
    ----------
    See base class.
    idx : int
        The index of the individual variable in the vector of binary variables
        to use as the binary variable for the implication.
    inequality : constraint.InequalityConstraint
        The inequality constraint to be satisfied, conditioned on the
        implication. This is assumed to not be a binary or integer constraint.
    big_m_vec : function
        A function of time that returns a vector (numpy.ndarray) of values such
        that, when subtracted from the original inequality constraint portion,
        the inequality is satisfied for all state and control values in the
        region of interest. Typically consists of large, positive valued
        elements. This implicitly requires that the returned vector is the same
        size as `bound`.

    Parameters
    ----------
    See base class.
    idx : int
        The index of the individual variable in the vector of binary variables
        to use as the binary variable for the implication.
    inequality : constraint.InequalityConstraint
        The inequality constraint to be satisfied, conditioned on the
        implication. This is assumed to not be a binary or integer constraint.
    big_m_vec : function
        A function of time that returns a vector (numpy.ndarray) of values such
        that, when subtracted from the original inequality constraint portion,
        the inequality is satisfied for all state and control values in the
        region of interest. Typically consists of large, positive valued
        elements. This implicitly requires that the returned vector is the same
        size as `bound`.

    """

    def __init__(self, idx, inequality, big_m_vec, n_binary):  # noqa: D107\
        super().__init__(n_binary,
                         inequality.n_state,
                         inequality.n_ctrl,
                         inequality.bound)
        self.idx = idx
        self.inequality = inequality
        self.big_m_vec = big_m_vec

    def constraint(self, t, state, ctrl, binary):
        """
        Return the constraint value of the re-formulated big-M constraint.

        Parameters
        ----------
        See base class.

        Returns
        -------
        See base class.

        """
        big_m = self.big_m_vec(t)
        ineq_val = self.inequality.constraint(t, state, ctrl)
        selection_vec = np.zeros((self.n_binary,), dtype=np.bool)
        selection_vec[self.idx] = True

        bin_val = selection_vec @ binary

        return ineq_val + (bin_val - 1) * big_m
