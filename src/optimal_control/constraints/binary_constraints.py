"""Represent constraints involving binary variables."""


import numpy as np

from optimal_control.constraints import constraint
import optimal_control.constraints.linear_constraints as lin_constr
import optimal_control.sparse_utils as sparse


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
            super().__init__(implications_linear.n_binary,
                             self.inequality.n_state, self.inequality.n_ctrl,
                             self.inequality.bound)
        else:
            if not all([ineq_mats, b_vec, m_mat, sel_mat]):
                raise TypeError("If `implications_linear` is not specified, "
                                "all of the other arguments must be given a "
                                "value.")
            a_mat_0, b_mat_0 = ineq_mats(0)
            n_state = a_mat_0.shape[0]
            n_ctrl = b_mat_0.shape[1]
            n_binary = sel_mat.shape[1]
            super().__init__(n_binary, n_state, n_ctrl, b_vec)
            self.inequality = lin_constr.LinearInequalityConstraint(ineq_mats,
                                                                    self.bound)
            self.m_mat = m_mat

    def _check_dimensional_consistency(self, implications):
        dims = [[imp.n_state, imp.n_ctrl, imp.n_binary] for imp in implications]
        n_state = dims[:][0]
        n_ctrl = dims[:][1]
        n_binary = dims[:][2]
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
        imps = [[imp.inequality.ineq_mats, imp.inequality.bound, imp.big_m_vec,
                 imp.idx]
                for imp in implications]

        ineq_mats = imps[:][0]
        bounds = imps[:][1]
        big_m_vecs = imps[:][2]
        idxs = imps[:][3]
        selection_mat = np.zeros((len(idxs), implications[0].n_binary))
        for i, idx in enumerate(idxs):
            selection_mat[i, idx] = 1

        def _aggregate_mats(t, mats):
            a_b_mats = [mat(t) for mat in mats]
            return sparse.block_diag(a_b_mats[:][0]), sparse.block_diag(a_b_mats[:][1])

        def _aggregate_vecs(t, vecs):
            return np.concatenate([vec(t) for vec in vecs])

        return (lin_constr_ineq(lambda t: _aggregate_mats(t, ineq_mats),
                                lambda t: _aggregate_vecs(t, bounds)),
                lambda t: _aggregate_mats(t, big_m_vecs),
                selection_mat)

    def constraint(self, t, state, ctrl, binary):
        """See base class."""
        bin_vars = self.selection_mat @ binary
        ineq_val = self.inequality.constraint(t, state, ctrl)
        m_mat = self.m_mat(t)

        return ineq_val + m_mat @ (bin_vars - 1)


class BinaryLinearEqualityConstraint(BinaryEqualityConstraint):
    """
    Represent a constraint linear in state, control, and binary variables.

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
