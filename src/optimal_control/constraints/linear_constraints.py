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

    """
    def __init__(self, A_eq_list=(), b_eq_list=(), A_ineq_list=(),
                 b_ineq_list=(), eps=1e-6):
        if (len(A_eq_list) != len(b_eq_list)
                or len(A_ineq_list) != len(b_ineq_list)):
            raise TypeError("The lengths of the respective lists of A matrices "
                            "and b vectors must be consistent for both the "
                            "equality and inequality lists.")
        eq_consts = [LinearEqualityConstraint(A_eq, b_eq, eps)
                     for A_eq, b_eq in zip(A_eq_list, b_eq_list)]
        ineq_consts = [LinearInequalityConstraint(A_ineq, b_ineq)
                       for A_ineq, b_ineq in zip(A_ineq_list, b_ineq_list)]
        super(LinearConstraints, self).__init__(eq_consts, ineq_consts)

    def equality_mat_vec(self, as_sparse=True):
        """
        Return the A and b of Ax = b from stacking the equality constraints.

        Parameters
        ----------
        as_sparse : bool, optional
            Indicate whether to return a sparse or dense format A matrix. Sparse
            (True) is the default.

        Returns
        -------
        A : scipy.sparse.spmatrix or numpy.ndarray
            The A matrix of the stacked Ax = b constraint.
        b : numpy.ndarray
            The b vector of the stacked Ax = b constraint.

        """
        eq_mats = [scipy.sparse.csc_matrix(const.A)
                   for const in self.eq_constraints]
        eq_vecs = [const.eq_val for const in self.eq_constraints]

        if as_sparse:
            return scipy.sparse.vstack(eq_mats), np.concatenate(eq_vecs)
        else:
            return (scipy.sparse.vstack(eq_mats).toarray(),
                    np.concatenate(eq_vecs))

    def inequality_mat_vec(self, as_sparse=True):
        """
        Return the A and b of Ax <= b from stacking the inequality constraints.

        Parameters
        ----------
        as_sparse : bool, optional
            Indicate whether to return a sparse or dense format A matrix. Sparse
            (True) is the default.

        Returns
        -------
        A : scipy.sparse.spmatrix or numpy.ndarray
            The A matrix of the stacked Ax <= b constraint.
        b : numpy.ndarray
            The b vector of the stacked Ax <= b constraint.

        """
        ineq_mats = [scipy.sparse.csc_matrix(const.A)
                   for const in self.eq_constraints]
        ineq_vecs = [const.eq_val for const in self.eq_constraints]

        if as_sparse:
            return scipy.sparse.vstack(ineq_mats), np.concatenate(ineq_vecs)
        else:
            return (scipy.sparse.vstack(ineq_mats).toarray(),
                    np.concatenate(ineq_vecs))


class LinearEqualityConstraint(constraint.EqualityConstraint):
    """
    Represent a linear equality constraint.

    Attributes
    ----------
    A : func
        A function of time that returns the A matrix of a linear equality
        constraint Ay = b.
    eq_val : func
        A function of time that returns the b vector of a linear equality
        constraint Ay = b.
    eps : double
        The tolerance of the equality.

    Parameters
    ----------
    A : func
        A function of time that returns the A matrix of a linear equality
        constraint Ay = b.
    b : func
        A function of time that returns the the b vector of a linear equality
        constraint Ay = b.
    eps : double, optional
        The tolerance of the equality.

    """
    def __init__(self, A, b, eps=1e-6):
        super(LinearEqualityConstraint, self).__init__(b, eps)
        self.A = A

    def constraint(self, t, state, ctrl):
        y = np.concatenate([state, ctrl])
        return np.dot(self.A(t), y)


class LinearInequalityConstraint(constraint.InequalityConstraint):
    """
    Represent a linear inequality constraint.

    Represents linear inequalities with upper bounds only. No generality is lost
    with this restriction as both sides of a lower bounded inequality can be
    negated to achieve an upper bound constraint.

    Attributes
    ----------
    A : func
        A function of time that returns the A matrix of Ay <= b.
    bound : func
        A function of time that returns the b vector of Ay <= b.

    Parameters
    ----------
    A : func
        A function of time that returns the A matrix of Ay <= b.
    b : func
        A function of time that returns the b vector of Ay <= b.

    """
    def __init__(self, A, b):
        super(LinearInequalityConstraint, self).__init__(upper_bound=b)
        self.A = A

    def constraint(self, t, state, ctrl):
        y = np.concatenate((state, ctrl))
        return np.dot(self.A(t), y)
