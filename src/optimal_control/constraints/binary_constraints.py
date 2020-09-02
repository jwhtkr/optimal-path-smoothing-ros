"""Represent constraints involving binary variables."""

import collections

import numpy as np

from optimal_control.constraints import constraint
import optimal_control.constraints.linear_constraints as lin_constr
import optimal_control.sparse_utils as sparse


def any_none(iterable):
    is_none = [item is None for item in iterable]
    return any(is_none)


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


class ImplicationInequalityAggregateConstraint(BinaryInequalityConstraint):
    """
    Represent an aggregated implication constraint.

    Where `ImplicationInequalityConstraint` is a single implication (nice for
    defining constraints, and representing general implied constraints), this
    class represents a set of implications where all the constraints are
    linear inequalities and so can all be aggregated together into one.

    Attributes
    ----------
    See base class for additional attributes.
    inequality : optimal_control.constraints.linear_constraints.LinearInequalityConstraint
        The aggregated linear inequality constraint.
    m_mat : function
        A function of time that returns a matrix (typically
        scipy.sparse.spmatrix sparse diagonal) of the aggregated "big-M" vectors
        of the implications.
    selection_mat : scipy.sparse.spmatrix
        A selection matrix that selects the applicable binary variables from the
        vector of binary variables.


    Parameters
    ----------
    See base class for additional parameters.
    implications_linear : list of ImplicationInequalityConstraint, optional
        The implications with linear inequality constraints that are to be
        aggregated. If `a_mat`, `b_vec`, `m_mat`, and `sel_mat` are not
        specified, then `implications_linear` must be specified.
    ineq_mats : function, optional
        A function of time that returns the A and B matrices of the linear
        inequality constraint that is already aggregated. Ignored if
        `implications_linear` specified. Required if `implications_linear` is
        not given.
    b_vec : function, optional
        A function of time that returns the b vector of the linear inequality
        constraint that is already aggregated. Ignored if `implications_linear`
        is specified. Required if `implications_linear` is not given.
    m_mat : function, optional
        A function of time that returns the big-M matrix for the implication.
        Ignored if `implications_linear` is specified. Required if
        `implications_linear` is not given.
    sel_mat : scipy.sparse.spmatrix
        The matrix that selects the binary variables for the implication.
        Ignored if `implications_linear` is specified. Required if
        `implications_linear` is not given.
    """

    def __init__(self, implications_linear=None, ineq_mats=None, b_vec=None,
                 m_mat=None, sel_mat=None):  # noqa: D107
        if implications_linear:
            self._check_dimensional_consistency(implications_linear)  # Raises ValueError on fail
            ineq, m_mat, sel_mat = self._aggregate(implications_linear)
            self.inequality, self.m_mat, self.selection_mat = ineq, m_mat, sel_mat
            super().__init__(sel_mat.shape[1],
                             self.inequality.n_state, self.inequality.n_ctrl,
                             self.inequality.bound)
        else:
            if any_none([ineq_mats, b_vec, m_mat, sel_mat]):
                raise TypeError("If `implications_linear` is not specified, "
                                "all of the other arguments must be given a "
                                "value.")
            a_mat_0, b_mat_0 = ineq_mats(0)
            n_state = a_mat_0.shape[1]
            n_ctrl = b_mat_0.shape[1]
            n_binary = sel_mat.shape[1]
            super().__init__(n_binary, n_state, n_ctrl, b_vec)
            self.inequality = lin_constr.LinearInequalityConstraint(ineq_mats,
                                                                    self.bound)
            self.m_mat = m_mat
            self.selection_mat = sel_mat

    def _check_dimensional_consistency(self, implications):
        n_state = [imp.n_state for imp in implications]
        n_ctrl = [imp.n_ctrl for imp in implications]
        n_binary = [imp.n_binary for imp in implications]
        if not n_state.count(n_state[0]) == len(n_state):
            raise ValueError("All of the implication constraints must be for "
                             "the same number of states.")
        if not n_ctrl.count(n_ctrl[0]) == len(n_ctrl):
            raise ValueError("All of the implication constraints must be for "
                             "the same number of controls")
        if not n_binary.count(n_binary[0]) == len(n_binary):
            raise ValueError("All of the implication constraints must be for "
                             "the same number of binary variables.")

    def _aggregate(self, implications):
        lin_constr_ineq = lin_constr.LinearInequalityConstraint

        ineq_mats = [imp.inequality.ineq_mats for imp in implications]
        bounds = [imp.bound for imp in implications]
        big_m_vecs = [imp.big_m_vec for imp in implications]
        idxs = [imp.idx for imp in implications]

        selection_mat = np.zeros((len(idxs), implications[0].n_binary))

        for i, idx in enumerate(idxs):
            selection_mat[i, idx] = 1

        def _aggregate_a_b_mats(t, mats):
            a_b_mats = [mat(t) for mat in mats]
            a_mats = [sparse.coo_matrix(a_b[0]) for a_b in a_b_mats]
            b_mats = [sparse.coo_matrix(a_b[1]) for a_b in a_b_mats]
            return sparse.vstack(a_mats), sparse.vstack(b_mats)

        def _aggregate_b_vecs(t, vecs):
            return np.concatenate([vec(t) for vec in vecs])

        def _aggregate_m_vecs(t, vecs):
            return sparse.block_diag([vec(t) for vec in vecs])

        return (lin_constr_ineq(lambda t: _aggregate_a_b_mats(t, ineq_mats),
                                lambda t: _aggregate_b_vecs(t, bounds)),
                lambda t: _aggregate_m_vecs(t, big_m_vecs),
                selection_mat)

    def get_individual_rows(self, t):
        """Get the list of individual rows of implication constraints at `t`."""
        # imp_rows = []
        ImplicationRow = collections.namedtuple("ImplicationRow",
                                                ["a_row",
                                                 "b_row",
                                                 "rhs_val",
                                                 "bin_idx"])
        m_mat = self.m_mat(t)
        a_mat, b_mat = self.inequality.ineq_mats(t)
        a_mat, b_mat = a_mat.tocsc(), b_mat.tocsc()
        b_vec = self.bound(t)
        ms_mat = m_mat @ self.selection_mat

        for row, col in zip(*ms_mat.nonzero()):
            yield ImplicationRow(a_mat[row, :], b_mat[row, :], b_vec[row], col)

        # return imp_rows

    def constraint(self, t, state, ctrl, binary):
        """See base class."""
        fat_one = np.ones(binary.shape)
        ineq_val = self.inequality.constraint(t, state, ctrl)
        m_mat = self.m_mat(t)
        ms_mat = m_mat @ self.selection_mat

        return ineq_val + ms_mat @ binary - ms_mat @ fat_one


class BinaryLinearEqualityConstraint(BinaryEqualityConstraint):
    """
    Represent an equality constraint linear in the variables.

    The constraint value is determined with 3 matrices, A, B, and C as
    Ax + Bu + Ca, where x is state, u is control, and a is binary variables.

    Attributes
    ----------
    See base class for additional attributes.
    eq_mats : function
        A function of time that returns a tuple of matrices (either
        numpy.ndarray or scipy.sparse.spmatrix) that are, in order, the A, B,
        and C matrices of the constraint value Ax + Bu + Ca.


    Parameters
    ----------
    See base class for additional parameters.
    eq_mats : function
        A function of time that returns a tuple of matrices (either
        numpy.ndarray or scipy.sparse.spmatrix) that are, in order, the A, B,
        and C matrices of the constraint value Ax + Bu + Ca.

    """

    def __init__(self, eq_mats, eq_val, eps=1e-6):  # noqa: D107
        a_mat, b_mat, c_mat = eq_mats(0)
        n_binary = c_mat.shape[1]
        n_state = a_mat.shape[1]
        n_ctrl = b_mat.shape[1]
        super().__init__(n_binary, n_state, n_ctrl, eq_val, eps)
        self.eq_mats = eq_mats

    def constraint(self, t, state, ctrl, binary):
        """See base class."""
        a_mat, b_mat, c_mat = self.eq_mats(t)
        return a_mat @ state + b_mat @ ctrl + c_mat @ binary

class BinaryLinearInequalityConstraint(BinaryInequalityConstraint):
    """
    Represent an inequality constraint linear the variables.

    The constraint value is determined with 3 matrices, A, B, and C as
    Ax + Bu + Ca, where x is state, u is control, and a is binary variables.

    Attributes
    ----------
    See base class for additional attributes.
    ineq_mats : function
        A function of time that returns a tuple of matrices (either
        numpy.ndarray or scipy.sparse.spmatrix) that are, in order, the A, B,
        and C matrices of the constraint value Ax + Bu + Ca.


    Parameters
    ----------
    See base class for additional parameters.
    eq_mats : function
        A function of time that returns a tuple of matrices (either
        numpy.ndarray or scipy.sparse.spmatrix) that are, in order, the A, B,
        and C matrices of the constraint value Ax + Bu + Ca.

    """

    def __init__(self, ineq_mats, bound):  # noqa: D107
        a_mat, b_mat, c_mat = ineq_mats(0)
        n_binary = c_mat.shape[1]
        n_state = a_mat.shape[1]
        n_ctrl = b_mat.shape[1]
        super().__init__(n_binary, n_state, n_ctrl, bound)
        self.ineq_mats = ineq_mats

    def constraint(self, t, state, ctrl, binary):
        """See base class."""
        a_mat, b_mat, c_mat = self.ineq_mats(t)
        return a_mat @ state + b_mat @ ctrl + c_mat @ binary
