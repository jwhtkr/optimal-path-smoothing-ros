"""Contains classes to represent linear constraints."""

import optimal_control.sparse_utils as sparse
import numpy as np

from optimal_control.constraints import constraints, constraint


class LinearConstraints(constraints.Constraints):
    """
    Represent linear constraints.

    Represents a set of constraints that are all linear in nature. This can
    include direct bounds on individual state or control values with both upper
    and lower limits, as well as linear combinations of states and controls with
    both equality and inequality constraints. This class represents all
    inequality constraints as some form of A_ineq * y <= b_ineq. In other words,
    it assumes only a single-sided, upper-bounded inequality. Note that lower
    bounds can be converted to upper bounds by inverting the sign of both A_ineq
    and b_ineq.

    Attributes
    ----------
    eq_consts : list of LinearEqualityConstraint
        A list of linear equality constraints.
    ineq_consts : list of LinearInequalityConstraint
        A list of linear inequality constraints.

    Parameters
    ----------
    eq_mat_list : list of func, optional
        A list of functions that return the A matrices of Ay = b.
    eq_val_list : list of func, optional
        A list of functions that return the b vectors of Ay = b.
    eq_constraints : list of LinearEqualityConstraint, optional
        A list of pre-built `LinearEqualityConstraint` (or sub-class) objects.
    ineq_mat_list : list of func, optional
        A list of functions that return the A matrices of Ay <= b.
    ineq_bound_list : list of func, optional
        A list of functions that return the b vectors of Ay <= b.
    ineq_constraints : list of LinearInequalityConstraint, optional
        A list of pre-built `LinearInequalityConstraint` (or sub-class) objects.
    eps : double, optional
        The allowable tolerance for considering the equality constraint to be
        satisfied. Defaults to 1e-6.
    **kwargs
        Keyword arguments to be passed to the parent class.

    """

    def __init__(self, eq_mat_list=(), eq_val_list=(), eq_constraints=(),
                 ineq_mat_list=(), ineq_bound_list=(), ineq_constraints=(),
                 eps=1e-6, **kwargs): #noqa: D107
        if (len(eq_mat_list) != len(eq_val_list)
                or len(ineq_mat_list) != len(ineq_bound_list)):
            raise TypeError("The lengths of the respective lists of matrices "
                            "and value/bound vectors must be consistent for "
                            "both the equality and inequality lists.")
        if not all([isinstance(const, LinearEqualityConstraint)
                    for const in eq_constraints]):
            raise TypeError("All passed in pre-built `eq_constraints` must be "
                            "linear (instance or subclass of "
                            "`LinearEqualityConstraint`).")
        if not all([isinstance(const, LinearInequalityConstraint)
                    for const in ineq_constraints]):
            raise TypeError("All passed in pre-built `ineq_constraints` must be"
                            " linear (instance or subclass of "
                            "`LinearInequalityConstraint`).")

        eq_consts = [LinearEqualityConstraint(eq_mat, eq_val, eps)
                     for eq_mat, eq_val in zip(eq_mat_list, eq_val_list)]
        ineq_consts = [LinearInequalityConstraint(ineq_mat, ineq_bound)
                       for ineq_mat, ineq_bound in zip(ineq_mat_list,
                                                       ineq_bound_list)]
        if eq_constraints:
            eq_consts.extend(eq_constraints)
        if ineq_constraints:
            ineq_consts.extend(ineq_constraints)

        super(LinearConstraints, self).__init__(eq_consts, ineq_consts,
                                                **kwargs)

    def equality_mat_vec(self, t, as_sparse=True):
        """
        Return the joint equality constraints A, B, and b of Ax + Bu = b.

        Parameters
        ----------
        t : float
            The time at which to calculate A, B, and b.
        as_sparse : bool, optional
            Indicate whether to return a sparse or dense format A matrix. Sparse
            (True) is the default.

        Returns
        -------
        A : scipy.sparse.spmatrix or numpy.ndarray
            The A matrix of the stacked Ax + Bu = b constraint.
        B : scipy.sparse.spmatrix or numpy.ndarray
            The B matrix of the stacked Ax + Bu = b constraint.
        b : numpy.ndarray
            The b vector of the stacked Ax + Bu = b constraint.

        """
        eq_mats_a = []
        eq_mats_b = []
        for constr in self.eq_constraints:
            a_mat, b_mat = constr.eq_mats(t)
            if a_mat is not None and b_mat is not None:
                eq_mats_a.append(sparse.coo_matrix(a_mat))
                eq_mats_b.append(sparse.coo_matrix(b_mat))

        eq_vecs = [constr.eq_val(t) for constr in self.eq_constraints
                   if constr.eq_val(t) is not None]

        if not eq_mats_a or not eq_mats_b or not eq_vecs:
            return (sparse.coo_matrix((0, self.n_state)),
                    sparse.coo_matrix((0, self.n_ctrl)),
                    np.zeros((0, 1)))

        eq_mat_a = sparse.vstack(eq_mats_a)
        eq_mat_b = sparse.vstack(eq_mats_b)
        eq_vec = np.concatenate(eq_vecs)

        if as_sparse:
            return eq_mat_a, eq_mat_b, eq_vec
        return eq_mat_a.toarray(), eq_mat_b.toarray(), eq_vec

    def inequality_mat_vec(self, t, as_sparse=True):
        """
        Return the stacked inequality constraint's A, B, and b of Ax + Bu <= b.

        Parameters
        ----------
        t : float
            The time at which to calculate A, B, and b.
        as_sparse : bool, optional
            Indicate whether to return a sparse or dense format A and B
            matrices. Sparse (True) is the default.

        Returns
        -------
        A : scipy.sparse.spmatrix or numpy.ndarray
            The A matrix of the stacked Ax + Bu <= b constraint.
        B : scipy.sparse.spmatrix or numpy.ndarray
            The B matrix of the stacked Ax + Bu <= b constraint.
        b : numpy.ndarray
            The b vector of the stacked Ax + Bu <= b constraint.

        """
        ineq_mats_a = []
        ineq_mats_b = []
        for constr in self.ineq_constraints:
            a_mat, b_mat = constr.ineq_mats(t)
            ineq_mats_a.append(sparse.coo_matrix(a_mat))
            ineq_mats_b.append(sparse.coo_matrix(b_mat))

        ineq_vecs = [constr.bound(t) for constr in self.ineq_constraints]

        if not ineq_mats_a or not ineq_mats_b or not ineq_vecs:
            return (sparse.coo_matrix((0, self.n_state)),
                    sparse.coo_matrix((0, self.n_ctrl)),
                    np.zeros((0, 1)))

        ineq_mat_a = sparse.vstack(ineq_mats_a)
        ineq_mat_b = sparse.vstack(ineq_mats_b)

        ineq_vec = np.concatenate(ineq_vecs)
        if as_sparse:
            return ineq_mat_a, ineq_mat_b, ineq_vec
        return ineq_mat_a.toarray(), ineq_mat_b.toarray(), ineq_vec


class LinearEqualityConstraint(constraint.EqualityConstraint):
    """
    Represent a linear equality constraint.

    Attributes
    ----------
    eq_mats : func
        A function of time that returns the A and B matrices of a linear
        equality constraint Ax + Bu = b as the tuple (A, B).
    eq_val : func
        A function of time that returns the b vector of a linear equality
        constraint Ax + Bu = b.
    eps : double
        The tolerance of the equality.

    Parameters
    ----------
    eq_mats : func
        A function of time that returns the A and B matrices of a linear
        equality constraint Ax + Bu = b as the tuple (A, B).
    b : func
        A function of time that returns the the b vector of a linear equality
        constraint Ax + Bu = b.
    eps : double, optional
        The tolerance of the equality. The default value is 1e-6

    """

    def __init__(self, eq_mats, eq_val, eps=1e-6): # noqa: D107
        a_mat, b_mat = eq_mats(0)
        n_state = a_mat.shape[1]
        n_ctrl = b_mat.shape[1]
        super(LinearEqualityConstraint, self).__init__(n_state, n_ctrl, eq_val,
                                                       eps)
        self.eq_mats = eq_mats

    def constraint(self, t, state, ctrl):
        """See base class."""
        a_mat, b_mat = self.eq_mats(t)
        # return np.dot(a_mat, state) + np.dot(b_mat, ctrl)
        return a_mat @ state + b_mat @ ctrl


class LinearInequalityConstraint(constraint.InequalityConstraint):
    """
    Represent a linear inequality constraint.

    Represents linear inequalities with upper bounds only. No generality is lost
    with this restriction as both sides of a lower bounded inequality can be
    negated to achieve an upper bound constraint.

    Attributes
    ----------
    ineq_mats : func
        A function of time that returns the A and B matrices of Ax + Bu <= b.
        Returns a tuple of (A, B)
    bound : func
        A function of time that returns the b vector of Ax + Bu <= b.

    Parameters
    ----------
    ineq_mats : func
        A function of time that returns the A and B matrices of Ax + Bu <= b.
        Returns a tuple of (A, B)
    bound : func
        A function of time that returns the b vector of Ax + Bu <= b.

    """

    def __init__(self, ineq_mats, bound): # noqa: D107
        a_mat, b_mat = ineq_mats(0)
        n_state = a_mat.shape[1]
        n_ctrl = b_mat.shape[1]
        super(LinearInequalityConstraint, self).__init__(n_state, n_ctrl,
                                                         upper_bound=bound)
        self.ineq_mats = ineq_mats

    def constraint(self, t, state, ctrl):
        """See base class."""
        a_mat, b_mat = self.ineq_mats(t)
        # return np.dot(a_mat, state) + np.dot(b_mat, ctrl)
        return a_mat @ state + b_mat @ ctrl


class LinearTimeInstantConstraint(LinearEqualityConstraint):
    """
    Represent a terminal constraint that is linear in the end state and control.

    The A and B matrices of the terminal constraint Ax_f + Bu_f = b are used,
    along with zero matrices for any other time to create the time dependent
    constraint.

    Attributes
    ----------
    eq_mats_final : tuple of numpy.ndarray
        A tuple of (A, B) the A and B matrices of the linear terminal constraint
        Ax_f + Bu_f = b
    eq_val : numpy.ndarray
        The b vector of the linear terminal constraint Ax_f + Bu_f = b.
    eps : double
        The tolerance of the equality.

    Parameters
    ----------
    t_final : float
        The final time at which the constraint is applicable.
    eq_mats_final : tuple of numpy.ndarray
        A tuple of (A, B) the A and B matrices of the linear terminal constraint
        Ax_f + Bu_f = b
    eq_val : func
        A function of time that returns the b vector of a linear terminal
        constraint Ax_f + Bu_f = b.
    eps : double, optional
        The tolerance of the equality. The default value is 1e-6.
    """

    def __init__(self, t_inst, eq_mats_inst, eq_val_inst, eps=1e-6): # noqa: D107
        zero_mats = (np.empty((0, eq_mats_inst[0].shape[1])),
                     np.empty((0, eq_mats_inst[1].shape[1])))
        def eq_mats(t):
            """Return `eq_mats_inst` if `t_inst`, otherwise zero matrices."""
            if abs(t-t_inst) <= eps:
                return eq_mats_inst
            return zero_mats

        def eq_val(t):
            """Return `eq_val_inst` if `t_inst`, otherwise a zero vector."""
            if abs(t-t_inst) <= eps:
                return eq_val_inst
            return np.zeros((0, 1))

        super(LinearTimeInstantConstraint, self).__init__(eq_mats, eq_val, eps)
