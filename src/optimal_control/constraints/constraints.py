"""Contains classes to hold and manipulate multiple optimization constraints."""


import numpy as np


class Constraints(object):
    """
    Represent a set of constraints.

    Represents constraints as two distinct groups: one of equality constraints
    and another of inequality constraints. These can then be manipulated,
    inspected, or evaluated all at once.

    Attributes
    ----------
    eq_constraints : list of EqualityConstraint
        A list of equality constraints for an optimization problem.
    ineq_constraints : list of InequalityConstraint
        A list of inequality constraints for an optimization problem.


    Parameters
    ----------
    eq_constraints : list of EqualityConstraint, optional
        A list of equality constraints for an optimization problem. If not
        specified, then `ineq_constraints` must be specified.
    ineq_constraints : list of InequalityConstraint, optional
        A list of inequality constraints for an optimization problem. If not
        specified, then `eq_constraints` must be specified.
    constraints : Constraints, optional
        Another :obj:`Constraints` object from which the constraints are to be
        inherited. In other words, the constraints from `constraints` will be
        appended to `eq_constraints` and `ineq_constraints`.

    """

    def __init__(self, eq_constraints=(), ineq_constraints=(),
                 constraints=None): # noqa: D107
        if not eq_constraints and not ineq_constraints and not constraints:
            raise ValueError("At least one of `eq_constraints`, "
                             "`ineq_constraints`, or `constraints` must be "
                             "specified.")

        self.eq_constraints = list(eq_constraints)
        self.ineq_constraints = list(ineq_constraints)

        if constraints is not None:
            self.eq_constraints.extend(constraints.eq_constraints)
            self.ineq_constraints.extend(constraints.ineq_constraints)

        n_state = []
        n_ctrl = []
        for const in self.eq_constraints + self.ineq_constraints:
            n_state.append(const.n_state)
            n_ctrl.append(const.n_ctrl)
        # if all values of n_state and n_ctrl are not consistent it's a problem.
        if (not n_state.count(n_state[0]) == len(n_state)
                or not n_ctrl.count(n_ctrl[0]) == len(n_ctrl)):
            raise ValueError("All constraints must be for the same number of "
                             "states and controls.")
        else:
            self.n_state = n_state[0]
            self.n_ctrl = n_ctrl[0]

    def is_satisfied(self, t, state, ctrl):
        """
        Determine if the constraints are satisfied.

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
            A boolean indication if all of the constraints are satisfied.

        """
        vals = []
        for eq_const in self.eq_constraints:
            vals.append(eq_const.is_satisfied(t, state, ctrl))
        for ineq_const in self.ineq_constraints:
            vals.append(ineq_const.is_satisfied(t, state, ctrl))

        return all(vals)

    def constraints(self, t, state, ctrl):
        """
        Calculate the values of the constraints.

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
            A vector of the constraint values calculated from each of the
            constraints. Ordered as equality constraints first and then
            inequality constraints. Returns as a vector of shape
            (len(n_eq_constraints)+len(n_ineq_constraints),1).

        """
        vals = np.empty(0)
        for eq_const in self.eq_constraints:
            np.concatenate([vals, eq_const.constraint(t, state, ctrl)])
        for ineq_const in self.ineq_constraints:
            np.concatenate([vals, ineq_const.constraint(t, state, ctrl)])

        return vals.reshape(-1, 1)

    # def derivatives(self, t, state, ctrl):
    #     """
    #     Calculate the derivatives of the constraints at a point.

    #     The derivatives are calculated both with respect to the state and the
    #     control input

    #     Parameters
    #     ----------
    #     t : double
    #         The time at which to evalutate the constraint.
    #     state : numpy.ndarray
    #         The state at time `t` for which the constraint should be evaluated.
    #     ctrl : numpy.ndarray
    #         The control input at time `t` for which the constraint should be
    #         evalutated.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         A vector of the constraint derivatives calculated from each of the
    #         constraints with respect to state. Ordered with equality constraints
    #         first and then inequality constraints. Returns as an array of shape
    #         (len(eq_constraints)+len(ineq_constraints), len(state)).
    #     numpy.ndarray
    #         A vector of the constraint derivatives calculated from each of the
    #         constraints with respect to control input. Ordered with equality
    #         constraints first and then inequality constraints. Returns as an
    #         array of shape
    #         (len(eq_constraints)+len(ineq_constraints), len(ctrl)).
    #     """
    #     vals = []
    #     for eq_const in self.eq_constraints:
    #         vals.append(eq_const.derivative(t, state, ctrl))
    #     for ineq_const in self.ineq_constraints:
    #         vals.append(ineq_const.derivative(t, state, ctrl))

    #     return np.array(vals, ndmin=2).T
