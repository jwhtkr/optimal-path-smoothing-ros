"""Contains the base object for a dynamic system."""


class Dynamics(object):  # pylint: disable=too-few-public-methods
    """
    Represent a dynamical system.

    Represent a mathematical model for a system in state space form. Tools for
    visualizing and manipulating the system are also included.

    Attributes
    ----------
    parameters : dict
        A dictionary of system parameters.
    n_state : int
        The number of states of the system.
    n_ctrl : int
        The number of control inputs of the system.
    n_output : int
        The number of outputs of the system.

    Parameters
    ----------
    n_state : int
        The number of states of the system.
    n_ctrl : int
        The number of control inputs of the system.
    n_output : int
        The number of outputs of the system.
    parameters : dict
        A dictionary of system parameters.

    """
    def __init__(self, n_state, n_ctrl, n_output, parameters):
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_output = n_output
        self.parameters = parameters


class ContinuousDynamics(Dynamics):
    """
    Represent a time-continuous dynamic system.

    Represent a dynamical system that is continuous in time. It may be either
    linear or non-linear, as well as time-varying or time-invariant. The
    dynamics are represented by an instantaneous time-derivative of the state
    given a control input at a particular time.

    """

    def state_derivative(self, t, state, ctrl):
        """
        Calculate the dynamics of the system at particular point as an np array.

        Calculate the derivative of the state, or the dynamic relationship, for
        a given time, state, and control input. Calculated according to the
        canonical formulation state_derivative = f(t, state, ctrl). This
        function must be overidden in child classes.

        Parameters
        ----------
        t : double
            The time at which to calculate the dynamics.
        state : np.array
            The state at time t for which the dynamics are to be calculated.
            Represented as a vector of shape (n_state,1).
        ctrl : np.array
            The control input at time t for which the dynamics are to be
            calculated. Represented as a vector of shape (n_ctrl,1).

        Returns
        -------
        numpy.ndarray
            The derivative of the state according to the dynamic relationship.
            Returns as a vector of shape (n_state,1).

        """
        raise NotImplementedError

    def output(self, t, state, ctrl):
        """
        Calculate the output of the system at a point as an numpy array.

        Calculate the output of the system for a given time, state, and control
        input. This relates to the canonical representation of the output:
        output = g(t, state, ctrl). Must be overidden by child classes.

        Parameters
        ----------
        t : double
            The time at which to calculate the dynamics.
        state : np.array
            The state at time t for which the dynamics are to be calculated.
            Represented as a vector of shape (n_state,1).
        ctrl : np.array
            The control input at time t for which the dynamics are to be
            calculated. Represented as a vector of shape (n_ctrl,1).

        Returns
        -------
        numpy.ndarray
            The output at time t, for the state state and control input ctrl as
            a vector of shape (n_output,1).

        """
        raise NotImplementedError

    def linearize(self, state_star, ctrl_star):
        """
        Linearize the dynamics about a point (state_star, ctrl_star).

        Linearize the system about the given state, state_star, and control
        input, ctrl_star. Returns the A, B, C, and D matrices as functions of
        time. Must be overidden by the child class.

        Parameters
        ----------
        state_star : numpy.ndarray
            The state about which to linearize the system. Represented as a
            vector of shape (n_state,1).
        ctrl_star : numpy.ndarray
            The control input about which to linearize the system. Represented
            as a vector of shape (n_ctrl,1).

        Returns
        -------
        A : function
            The linear A matrix of state_dot = A(t)state + B(t)ctrl. It is
            returned as a function of time that can be called to return a numpy
            array of shape (n_state, n_state).
        B : function
            The linear B matrix of state_dot = A(t)state + B(t)ctrl. It is
            returned as a function of time that can be called to return a numpy
            array of shape (n_state, n_ctrl).
        C : function
            The linear C matrix of output = C(t)state + D(t)ctrl. It is
            returned as a function of time that can be called to return a numpy
            array of shape (n_output,n_state).
        D : function
            The linear D matrix of output = C(t)state + D(t)ctrl. It is
            returned as a function of time that can be called to return a numpy
            array of shape (n_output,n_ctrl).

        """
        raise NotImplementedError

    def discretize(self, time_step, method='euler'):
        """
        Return an approximate ``DiscreteDynamics`` object.

        Build and return a ``DiscreteDynamics`` object that approximates the
        continuous dynamics with discrete dynamics for a given time step size.
        Must be overidden by the child class.

        Parameters
        ----------
        time_step : double
            The time step at which to discretize the system. This results in
            state(t+time_step) ~= f_k(state(t), ctrl(t)).
        method : {'euler', 'rk4'}
            One of [euler, rk4]. Indicates the method of discretization to use.
            Default is 'euler'. More methods will be added in the future.

        Returns
        -------
        DiscreteDynamics
            The ``DiscreteDynamics`` object that represents the discretized
            system. This is only valid for the given time step parameter.

        """
        raise NotImplementedError


class DiscreteDynamics(Dynamics):  # pylint: disable=too-few-public-methods
    """
    Represent a discrete or discretized system's dynamics.

    A discrete-time (or discretized continuous-time) system is represented. It
    can be non-linear or linear, and time-varying or time-invariant. The
    dynamics are represented with difference equations.

    Attributes
    ----------
    time_step : double
        The time step of the discretization. Corresponds to:
        t_next = t_curr + time_step.

    """
    def output(self, k, curr_state, curr_ctrl):
        """
        Calculate the output of the system at a point as an numpy array.

        Calculate the output of the system for a given time index, state, and
        control input. This relates to the canonical representation of the
        output: output = g(k, state, ctrl). Must be overidden by child classes.

        Parameters
        ----------
        k : int
            The time index at which to calculate the dynamics.
        curr_state : np.array
            The state at time `k` for which the dynamics are to be calculated.
            Represented as a vector of shape (`n_state`,1).
        curr_ctrl : np.array
            The control input at time `k` for which the dynamics are to be
            calculated. Represented as a vector of shape (`n_ctrl`,1).

        Returns
        -------
        numpy.ndarray
            The output at time `k`, for the state state and control input ctrl
            as a vector of shape (`n_output`,1).

        """
        raise NotImplementedError

    def next_state(self, k, curr_state, curr_ctrl):
        """
        Calculate the next state from the discretized dynamics.

        Calculate teh next state as a function of the current state, current
        control input and the time index, k. Must be overidden by the child
        class.

        Parameters
        ----------
        k : int
            The time index at which to calculate the discrete dynamic step.
        curr_state : numpy.ndarray
            The current state for which the dynamic step is to be calculated.
            Represented as a vector of shape (n_state,1).
        curr_ctrl : numpy.ndarray
            The current control input for which the dynamic step is to be
            calculated. Represented as a vector of shape (n_ctrl,1).

        Returns
        -------
        numpy.ndarray
            The next state resulting from the dynamic step. Represented as a
            vector of shape (n_state,1).

        """
        raise NotImplementedError
