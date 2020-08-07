"""Contains classes to represent quadratic costs."""


import numpy as np

from optimal_control.objectives import cost


class QuadraticCost(cost.Cost):
    """Represent a general quadratic cost."""
    pass


class ContinuousQuadraticCost(QuadraticCost, cost.ContinuousCost):
    """
    Represents a continuous cost quadratic in the state and input.

    Attributes
    ----------
    inst_state_cost : numpy.ndarray
        The instantaneous cost matrix for the state.
    inst_ctrl_cost : numpy.ndarray
        The instantaneous cost matrix for the control input.
    term_state_cost : numpy.ndarray
        The terminal cost matrix for the state.
    desired_state : func
        A function that outputs a desired state vector for a given time `t`.
    desired_ctrl : func
        A function that outputs a desired control input vector for a given time
        `t`.

    Parameters
    ----------
    inst_state_cost : numpy.ndarray
        The instantaneous cost matrix for the state.
    inst_ctrl_cost : numpy.ndarray
        The instantaneous cost matrix for the control input.
    term_state_cost : numpy.ndarray
        The terminal cost matrix for the state.
    desired_state : func, optional
        A function that outputs a desired state vector for a given time `t`.
    desired_ctrl : func, optional
        A function that outputs a desired control input vector for a given time
        `t`.

    """
    def __init__(self, inst_state_cost, inst_ctrl_cost, term_state_cost,
                 desired_state=None, desired_ctrl=None):
        self.inst_state_cost = inst_state_cost
        self.inst_ctrl_cost = inst_ctrl_cost
        self.term_state_cost = term_state_cost

        if not desired_state:
            desired_state = self._zero_state
        if not desired_ctrl:
            desired_ctrl = self._zero_ctrl

        self.desired_state = desired_state
        self.desired_ctrl = desired_ctrl

    def _zero_state(self, t):
        """
        Create a zero state vector.

        Returns
        -------
        numpy.ndarray
            A zero column vector of the size of the state vector.
        """
        del t
        return np.zeros((self.inst_state_cost.shape[0], 1))

    def _zero_ctrl(self, t):
        """
        Create a zero control input vector.

        Returns
        -------
        numpy.ndarray
            A zero column vector of the size of the control input vector.
        """
        del t
        return np.zeros((self.inst_ctrl_cost.shape[0], 1))

    def instantaneous(self, t, state, ctrl):
        """See base class."""
        desired_state = self.desired_state(t)
        desired_ctrl = self.desired_ctrl(t)
        state_err = state - desired_state
        ctrl_err = ctrl - desired_ctrl

        state_cost = np.dot(state_err.T,
                            np.dot(self.inst_state_cost, state_err))
        ctrl_cost = np.dot(ctrl_err.T, np.dot(self.inst_ctrl_cost, ctrl_err))

        return state_cost + ctrl_cost

    def terminal(self, t, state, ctrl):
        """See base class."""
        desired_state = self.desired_state(t)
        state_err = state - desired_state

        return np.dot(state_err.T, np.dot(self.term_state_cost, state_err))


class DiscreteQuadraticCost(QuadraticCost, cost.DiscreteCost):
    """
    Represents a discrete cost quadratic in the state and input.

    Attributes
    ----------
    inst_state_cost : numpy.ndarray
        The instantaneous cost matrix for the state.
    inst_ctrl_cost : numpy.ndarray
        The instantaneous cost matrix for the control input.
    term_state_cost : numpy.ndarray
        The terminal cost matrix for the state.
    desired_state : func
        A function that outputs a desired state vector for a given time index
        `k`.
    desired_ctrl : func
        A function that outputs a desired control input vector for a given time
        index `k`.

    Parameters
    ----------
    inst_state_cost : numpy.ndarray
        The instantaneous cost matrix for the state.
    inst_ctrl_cost : numpy.ndarray
        The instantaneous cost matrix for the control input.
    term_state_cost : numpy.ndarray
        The terminal cost matrix for the state.
    desired_state : func, optional
        A function that outputs a desired state vector for a given time index
        `k`.
    desired_ctrl : func, optional
        A function that outputs a desired control input vector for a given time
        index `k`.

    """
    def __init__(self, inst_state_cost, inst_ctrl_cost, term_state_cost,
                 desired_state=None, desired_ctrl=None):
        self.inst_state_cost = inst_state_cost
        self.inst_ctrl_cost = inst_ctrl_cost
        self.term_state_cost = term_state_cost

        if not desired_state:
            desired_state = self._zero_state
        if not desired_ctrl:
            desired_ctrl = self._zero_ctrl

        self.desired_state = desired_state
        self.desired_ctrl = desired_ctrl

    def _zero_state(self, k):
        """
        Create a zero state vector.

        Returns
        -------
        numpy.ndarray
            A zero column vector of the size of the state vector.
        """
        del k
        return np.zeros((self.inst_state_cost.shape[0], 1))

    def _zero_ctrl(self, k):
        """
        Create a zero control input vector.

        Returns
        -------
        numpy.ndarray
            A zero column vector of the size of the control input vector.
        """
        del k
        return np.zeros((self.inst_ctrl_cost.shape[0], 1))

    def instantaneous(self, k, state, ctrl):
        """See base class."""
        desired_state = self.desired_state(k)
        desired_ctrl = self.desired_ctrl(k)
        state_err = state - desired_state
        ctrl_err = ctrl - desired_ctrl

        # state_cost = np.dot(state_err.T,
        #                     np.dot(self.inst_state_cost, state_err))
        # ctrl_cost = np.dot(ctrl_err.T, np.dot(self.inst_ctrl_cost, ctrl_err))
        
        if isinstance(state_err, np.ndarray):
            state_cost = state_err.T @ self.inst_state_cost @ state_err
        else:
            state_cost = state_err @ self.inst_state_cost @ state_err
            
        if isinstance(ctrl_err, np.ndarray):
            ctrl_cost = ctrl_err.T @ self.inst_ctrl_cost @ ctrl_err
        else:
            ctrl_cost = ctrl_err @ self.inst_ctrl_cost @ ctrl_err
        
        return state_cost + ctrl_cost

    def terminal(self, k, state, ctrl):
        """See base class."""
        desired_state = self.desired_state(k)
        state_err = state - desired_state

        # return np.dot(state_err.T, np.dot(self.term_state_cost, state_err))
        if isinstance(state_err, np.ndarray):
            return state_err.T @ self.term_state_cost @ state_err
        return state_err @ self.term_state_cost @ state_err


class ContinuousCondensedQuadraticCost(ContinuousQuadraticCost):
    """
    Represents a continuous condensed quadratic cost to be minimized.

    The constant terms of the quadratic are dropped, resulting in a slightly
    condensed form that has the same minimizer as the full quadratic cost.

    Attributes
    ----------
    inst_state_cost : numpy.ndarray
        The instantaneous cost matrix for the state.
    inst_ctrl_cost : numpy.ndarray
        The instantaneous cost matrix for the control input.
    term_state_cost : numpy.ndarray
        The terminal cost matrix for the state.
    desired_state : func
        A function that outputs a desired state vector for a given time `t`.
    desired_ctrl : func
        A function that outputs a desired control input vector for a given time
        `t`.

    Parameters
    ----------
    inst_state_cost : numpy.ndarray
        The instantaneous cost matrix for the state.
    inst_ctrl_cost : numpy.ndarray
        The instantaneous cost matrix for the control input.
    term_state_cost : numpy.ndarray
        The terminal cost matrix for the state.
    desired_state : func
        A function that outputs a desired state vector for a given time `t`.
    desired_ctrl : func
        A function that outputs a desired control input vector for a given time
        `t`.

    """
    def instantaneous(self, t, state, ctrl):
        """See base class."""
        desired_state = self.desired_state(t)
        desired_ctrl = self.desired_ctrl(t)

        state_cost_quad = np.dot(state.T, np.dot(self.inst_state_cost, state))
        state_cost_lin = -2*np.dot(desired_state.T,
                                   np.dot(self.inst_state_cost, state))

        ctrl_cost_quad = np.dot(ctrl.T, np.dot(self.inst_ctrl_cost, ctrl))
        ctrl_cost_lin = -2*np.dot(desired_ctrl.T,
                                  np.dot(self.inst_ctrl_cost, ctrl))

        return state_cost_quad + state_cost_lin + ctrl_cost_quad + ctrl_cost_lin

    def terminal(self, t, state, ctrl):
        """See base class."""
        desired_state = self.desired_state(t)

        quad = np.dot(state.T, np.dot(self.term_state_cost, state))
        lin = -2*np.dot(desired_state.T, np.dot(self.term_state_cost, state))

        return quad + lin


class DiscreteCondensedQuadraticCost(DiscreteQuadraticCost):
    """
    Represents a discrete condensed quadratic cost to be minimized.

    The constant terms of the quadratic are dropped, resulting in a slightly
    condensed form that has the same minimizer as the full quadratic cost.

    Attributes
    ----------
    inst_state_cost : numpy.ndarray
        The instantaneous cost matrix for the state.
    inst_ctrl_cost : numpy.ndarray
        The instantaneous cost matrix for the control input.
    term_state_cost : numpy.ndarray
        The terminal cost matrix for the state.
    desired_state : func
        A function that outputs a desired state vector for a given time index
        `k`.
    desired_ctrl : func
        A function that outputs a desired control input vector for a given time
        index `k`.

    Parameters
    ----------
    inst_state_cost : numpy.ndarray
        The instantaneous cost matrix for the state.
    inst_ctrl_cost : numpy.ndarray
        The instantaneous cost matrix for the control input.
    term_state_cost : numpy.ndarray
        The terminal cost matrix for the state.
    desired_state : func, optional
        A function that outputs a desired state vector for a given time index
        `k`.
    desired_ctrl : func, optional
        A function that outputs a desired control input vector for a given time
        index `k`.

    """
    def instantaneous(self, k, state, ctrl):
        """See base class."""
        desired_state = self.desired_state(k)
        desired_ctrl = self.desired_ctrl(k)

        if isinstance(state, np.ndarray):
            # state_cost_quad = np.dot(state.T, np.dot(self.inst_state_cost, state))
        
            state_cost_quad = state.T @ self.inst_state_cost @ state
        else:
            state_cost_quad = state @ self.inst_state_cost @ state
            
        # state_cost_lin = -2*np.dot(desired_state.T,
        #                            np.dot(self.inst_state_cost, state))   
        state_cost_lin = -2 * desired_state.T @ self.inst_state_cost @ state

        if isinstance(ctrl, np.ndarray):
            # ctrl_cost_quad = np.dot(ctrl.T, np.dot(self.inst_ctrl_cost, ctrl))
            ctrl_cost_quad = ctrl.T @ self.inst_ctrl_cost @ ctrl
        else:
            ctrl_cost_quad = ctrl @ self.inst_ctrl_cost @ ctrl
            
        # ctrl_cost_lin = -2*np.dot(desired_ctrl.T,
        #                           np.dot(self.inst_ctrl_cost, ctrl))
        ctrl_cost_lin = -2 * desired_ctrl.T @ self.inst_ctrl_cost @ ctrl

        return state_cost_quad + state_cost_lin + ctrl_cost_quad + ctrl_cost_lin

    def terminal(self, k, state, ctrl):
        """See base class."""
        desired_state = self.desired_state(k)

        if isinstance(state, np.ndarray):
            # quad = np.dot(state.T, np.dot(self.term_state_cost, state))
            quad = state.T @ self.term_state_cost @ state
        else:
            quad = state @ self.term_state_cost @ state
        # lin = -2*np.dot(desired_state.T, np.dot(self.term_state_cost, state))
        lin = -2 * desired_state.T @ self.term_state_cost @ state

        return quad + lin
