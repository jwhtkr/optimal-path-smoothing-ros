"""Contains classes to represent linear constraints."""

import scipy.sparse
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
    A_eq : list of func, optional
        A list of functions that return the A matrices of Ay = b.
    b_eq : list of func, optional
        A list of functions that return the b vectors of Ay = b.
    A_ineq : list of func, optional
        A list of functions that return the A matrices of Ay <= b.
    b_ineq : list of func, optional
        A list of functions that return the b vectors of Ay <= b.
    eps : double, optional
        The allowable tolerance for considering the equality constraint to be
        satisfied. Defaults to 1e-6.
    **kwargs
        Keyword arguments to be passed to the parent class.

    """
    def __init__(self, eq_mat_list=(), eq_val_list=(), ineq_mat_list=(),
                 ineq_bound_list=(), eps=1e-6, **kwargs):
        if (len(eq_mat_list) != len(eq_val_list)
                or len(ineq_mat_list) != len(ineq_bound_list)):
            raise TypeError("The lengths of the respective lists of matrices "
                            "and value/bound vectors must be consistent for "
                            "both the equality and inequality lists.")
        eq_consts = [LinearEqualityConstraint(eq_mat, eq_val, eps)
                     for eq_mat, eq_val in zip(eq_mat_list, eq_val_list)]
        ineq_consts = [LinearInequalityConstraint(ineq_mat, ineq_bound)
                       for ineq_mat, ineq_bound in zip(ineq_mat_list,
                                                       ineq_bound_list)]
        super(LinearConstraints, self).__init__(eq_consts, ineq_consts,
                                                **kwargs)

    def equality_mat_vec(self, t, as_sparse=True):
        """
        Return the aggregate equality constraint's A, B, and b of Ax + Bu = b.

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
        eq_mats_A = []
        eq_mats_B = []
        for const in self.eq_constraints:
            A, B = const.eq_mats(t)
            eq_mats_A.append(scipy.sparse.csc_matrix(A))
            eq_mats_B.append(scipy.sparse.csc_matrix(B))

        eq_vecs = [const.eq_val(t) for const in self.eq_constraints]

        if eq_mats_A is None or eq_mats_B is None or eq_vecs is None:
            return None, None, np.zeros((0, 1))

        eq_mat_A = scipy.sparse.vstack(eq_mats_A)
        eq_mat_B = scipy.sparse.vstack(eq_mats_B)
        eq_vec = np.concatenate(eq_vecs)

        if as_sparse:
            return eq_mat_A, eq_mat_B, eq_vec
        return eq_mat_A.toarray(), eq_mat_B.toarray(), eq_vec

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
        ineq_mats_A = []
        ineq_mats_B = []
        for const in self.ineq_constraints:
            A, B = const.ineq_mats(t)
            ineq_mats_A.append(scipy.sparse.csc_matrix(A))
            ineq_mats_B.append(scipy.sparse.csc_matrix(B))

        ineq_vecs = [const.eq_val(t) for const in self.eq_constraints]

        if not ineq_mats_A or not ineq_mats_B or not ineq_vecs:
            return None, None, np.zeros((0, 1))

        ineq_mat_A = scipy.sparse.vstack(ineq_mats_A)
        ineq_mat_B = scipy.sparse.vstack(ineq_mats_B)

        ineq_vec = np.concatenate(ineq_vecs)
        if as_sparse:
            return ineq_mat_A, ineq_mat_B, ineq_vec
        return ineq_mat_A.toarray(), ineq_mat_B.toarray(), ineq_vec


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
        The tolerance of the equality.

    """
    def __init__(self, eq_mats, eq_val, eps=1e-6):
        super(LinearEqualityConstraint, self).__init__(eq_val, eps)
        self.eq_mats = eq_mats

    def constraint(self, t, state, ctrl):
        A, B = self.eq_mats(t)
        return np.dot(A, state) + np.dot(B, ctrl)


class LinearInequalityConstraint(constraint.InequalityConstraint):
    """
    Represent a linear inequality constraint.

    Represents linear inequalities with upper bounds only. No generality is lost
    with this restriction as both sides of a lower bounded inequality can be
    negated to achieve an upper bound constraint.

    Attributes
    ----------
    matrices : func
        A function of time that returns the A and B matrices of Ax + Bu <= b.
        Returns a tuple of (A, B)
    bound : func
        A function of time that returns the b vector of Ax + Bu <= b.

    Parameters
    ----------
    matrices : func
        A function of time that returns the A and B matrices of Ax + Bu <= b.
        Returns a tuple of (A, B)
    bound : func
        A function of time that returns the b vector of Ax + Bu <= b.

    """
    def __init__(self, matrices, bound):
        super(LinearInequalityConstraint, self).__init__(upper_bound=bound)
        self.matrices = matrices

    def constraint(self, t, state, ctrl):
        A, B = self.matrices(t)
        return np.dot(A, state) + np.dot(B, ctrl)
