"""Contains a base class for representing a cost function (to be minimized)."""


import scipy.integrate as integrate

from optimal_control.objectives import objective


class Cost(objective.Objective):
    """Represent an objective that is meant to be minimized, i.e., a cost."""


class ContinuousCost(Cost, objective.ContinuousObjective):  # pylint: disable=abstract-method
    """Represent a continuous cost function for minimization."""

    def cost(self, t_initial, t_final, state_func, ctrl_func):
        """
        Calculate the total cost of a state and control trajectory.

        Parameters
        ----------
        t_initial : double
            The initial time of the trajectory to be evaluated.
        t_final : double
            The final time of the trajectory to be evalutated.
        state_func : func
            The state trajectory as a function of time.
        ctrl_func : func
            The control input trajectory as a function of time.

        Returns
        -------
        double
            The cost of the state and control trajectories over the indicated
            time horizon.
        """

        def instantaneous_func(t):
            """Calculate the instantaneous cost at a time instant `t`."""
            self.instantaneous(t, state_func(t), ctrl_func(t))

        instantaneous = integrate.quad(instantaneous_func, t_initial, t_final)
        terminal = self.terminal(t_final,
                                 state_func(t_final),
                                 ctrl_func(t_final))

        return instantaneous + terminal


class DiscreteCost(Cost, objective.DiscreteObjective):  # pylint: disable=abstract-method
    """Represent a discrete cost function for minimization."""

    def cost(self, k_initial, k_final, state_traj, ctrl_traj):
        """
        Calculate the total cost of a state and control input trajectory.

        Parameters
        ----------
        k_initial : int
            The initial time index of the trajectory to be evalutated.
        k_final : int
            The final time index of the trajectory to be evaluated.
        state_traj : numpy.ndarray
            The discrete state trajectory to be evaluated. It is an array of
            shape (n_state, k_final-k_initial+1).
        ctrl_traj : numpy.ndarray
            The discrete control input trajectory to be evaluated. It is an
            array of shape (n_ctrl, k_final-k_initial+1).

        Returns
        -------
        double
            The cost of the state and control input trajectories over the span
            of time indices indicated.

        """
        instantaneous = 0
        for k in range(k_initial, k_final):
            instantaneous += self.instantaneous(k,
                                                state_traj[:, k],
                                                ctrl_traj[:, k])

        terminal = self.terminal(k_final,
                                 state_traj[:, k_final],
                                 ctrl_traj[:, k_final])

        return instantaneous + terminal
