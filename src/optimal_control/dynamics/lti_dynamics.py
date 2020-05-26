"""Contains classes that represent Linear, Time-Invariant systems."""


import scipy.linalg
import numpy as np

from optimal_control.dynamics import dynamics


def _infer_n_state(A, B, C):
    """
    Infer the number of states from `A`, `B`, or `C`.

    If any are None (or evaluate to False in a boolean context), one of the
    others is used. They are not checked for consistency as this should
    arise in other contexts. If all are None (or False), then an exception
    is raised.

    Parameters
    ----------
    A : numpy.ndarray or None
        The A matrix from which to infer the number of states, `n_state`.
    B : numpy.ndarray or None
        The B matrix from which to infer the number of states, `n_state`.
    C : numpy.ndarray or None
        The C matrix from which to infer the number of states, `n_state`.

    Returns
    -------
    int
        The number of states of the system.
    """
    if A is not None:
        return A.shape[0]
    if B is not None:
        return B.shape[0]
    if C is not None:
        return C.shape[1]
    raise TypeError("One of A, B, or C must be valid (not None/empty/etc.).")


def _infer_n_ctrl(B, D):
    """
    Infer the number of control inputs from `B` or `D`.

    If any are None (or evaluate to False in a boolean context), one of the
    others is used. They are not checked for consistency as this should
    arise in other contexts. If all are None (or False), then an exception
    is raised.

    Parameters
    ----------
    B : numpy.ndarray or None
        The B matrix from which to infer the number of control inputs, `n_ctrl`.
    D : numpy.ndarray or None
        The D matrix from which to infer the number of control inputs `n_ctrl`.

    Returns
    -------
    int
        The number of states of the system.
    """
    if B is not None:
        return B.shape[1]
    if D is not None:
        return D.shape[1]
    raise TypeError("One of A, B, or C must be valid (not None/empty/etc.).")


def _infer_n_output(C, D):
    """
    Infer the number of states from `A`, `B`, or `C`.

    If any are None (or evaluate to False in a boolean context), one of the
    others is used. They are not checked for consistency as this should
    arise in other contexts. If all are None (or False), then an exception
    is raised.

    Parameters
    ----------
    C : numpy.ndarray or None
        The C matrix from which to infer the number of outputs, `n_output`.
    D : numpy.ndarray or None
        The D matrix from which to infer the number of outputs, `n_output`.

    Returns
    -------
    int
        The number of states of the system.
    """
    if C is not None:
        return C.shape[0]
    if D is not None:
        return D.shape[0]
    raise TypeError("One of A, B, or C must be valid (not None/empty/etc.).")


class CLTI_Dynamics(dynamics.ContinuousDynamics):
    """
    Represent Continuous Linear Time-Inveriant (CLTI) dynamics.

    Attributes
    ----------
    A : numpy.ndarray
        The A matrix from x_dot = Ax + Bu.
    B : numpy.ndarray
        The B matrix from x_dot = Ax + Bu.
    C : numpy.ndarray
        The C matrix from y = Cx + Du.
    D : numpy.ndarray
        The D matrix from y = Cx + Du.
    n_state : int
        The size of the state vector. It is inferred from the shape of A, B, or
        C.
    n_ctrl : int
        The size of the control input vector. It is inferred from the shape of B
        or D.
    n_output : int
        The size of the output vector. It is inferred from the shape of C or D.

    Parameters
    ----------
    A : numpy.ndarray
        The A matrix from x_dot = Ax + Bu.
    B : numpy.ndarray, optional
        The B matrix from x_dot = Ax + Bu. If omitted, a zero matrix is assumed.
        Cannot be omitted if D is omitted.
    C : numpy.ndarray, optional
        The C matrix from y = Cx + Du. If omitted, a zero matrix is assumed.
        Cannot be omitted if D is omitted.
    D : numpy.ndarray, optional
        The D matrix from y = Cx + Du. If omitted, a zero matrix is assumed.
        Cannot be omitted if B or C is omitted.

    """
    def __init__(self, A=None, B=None, C=None, D=None):
        n_state = _infer_n_state(A, B, C)
        n_ctrl = _infer_n_ctrl(B, D)
        n_output = _infer_n_output(C, D)

        if A is None:
            A = np.zeros((n_state, n_state))
        if B is None:
            B = np.zeros((n_state, n_ctrl))
        if C is None:
            C = np.zeros((n_output, n_state))
        if D is None:
            D = np.zeros((n_output, n_ctrl))

        super(CLTI_Dynamics, self).__init__(n_state, n_ctrl, n_output, {})
        self.A, self.B, self.C, self.D = A, B, C, D

    def state_derivative(self, t, state, ctrl):
        """See base class."""
        del t  # time invariant
        return np.dot(self.A, state) + np.dot(self.B, ctrl)

    def output(self, t, state, ctrl):
        """See base class."""
        del t  # time invariant
        return np.dot(self.C, state)  + np.dot(self.D, ctrl)

    def linearize(self, state_star, ctrl_star):
        """See base class."""
        return self.A, self.B, self.C, self.D

    def discretize(self, time_step, method='euler'):
        """See base class."""
        if method == "rk4":
            raise NotImplementedError("Discretization with Runge-Kutta 4th "
                                      "order methods is not yet supported.")
        elif method == "euler":
            raise NotImplementedError("Discretization with Euler methods "
                                      "is not yet supported.")
        elif method == "exact":
            return self._discretize_exact(time_step)
        else:
            raise TypeError("`method` can only be 'euler' or 'rk4' (at least "
                            "currently).")

    def _discretize_exact(self, time_step):
        """Discretize the continuous dynamics with exact methods."""
        tmp = np.concatenate((self.A, self.B), axis=1)
        tmp = np.concatenate((tmp, np.zeros((self.n_ctrl,
                                             self.n_state+self.n_ctrl))))
        tmp = scipy.linalg.expm(tmp*time_step)  # pylint: disable=no-member
        A_bar = tmp[0:self.n_state, 0:self.n_state]
        B_bar = tmp[0:self.n_state, self.n_state:self.n_state+self.n_ctrl]

        return DLTI_Dynamics(time_step, A_bar, B_bar, self.C, self.D)


class DLTI_Dynamics(dynamics.DiscreteDynamics):
    """
    Represent Discrete Linear Time-Invariant (DLTI) dynamics.

    Attributes
    ----------
    A : numpy.ndarray
        The A matrix of the discrete dynamic equation x_{k+1} = Ax_k + Bu_k.
    B : numpy.ndarray
        The B matrix of the discrete dynamic equation x_{k+1} = Ax_k + Bu_k.
    C : numpy.ndarray
        The C matrix of the discrete dynamic equation y_k = Cx_k + Du_k.
    D : numpy.ndarray
        The D matrix of the discrete dynamic equation y_k = Cx_k + Du_k.
    time_step : double
        The time step of the discretization. Corresponds to:
        t_next = t_curr + time_step.
    n_state : int
        The size of the state vector. It is inferred from the shape of A, B, or
        C.
    n_ctrl : int
        The size of the control input vector. It is inferred from the shape of B
        or D.
    n_output : int
        The size of the output vector. It is inferred from the shape of C or D.

    Parameters
    ----------
    time_step : double
        The time step of the discretization. Corresponds to:
        t_next = t_curr + time_step.
    A_discrete : numpy.ndarray
        The A matrix of the discrete dynamic equation x_{k+1} = Ax_k + Bu_k.
    B_discrete : numpy.ndarray, optional
        The B matrix of the discrete dynamic equation x_{k+1} = Ax_k + Bu_k. If
        omitted it is assumed to be a zero matrix.
    C_discrete : numpy.ndarray, optional
        The C matrix of the discrete dynamic equation y_k = Cx_k + Du_k. If
        omitted it is assumed to be a zero matrix.
    D_discrete : numpy.ndarray, optional
        The D matrix of the discrete dynamic equation y_k = Cx_k + Du_k. If
        omitted it is assumed to be a zero matrix.

    """
    def __init__(self, time_step, A_discrete, B_discrete=None,
                 C_discrete=None, D_discrete=None):
        n_state = _infer_n_state(A_discrete, B_discrete, C_discrete)
        n_ctrl = _infer_n_ctrl(B_discrete, D_discrete)
        n_output = _infer_n_output(C_discrete, D_discrete)

        super(DLTI_Dynamics, self).__init__(n_state, n_ctrl, n_output, {})

        if B_discrete is None:
            B_discrete = np.zeros((n_state, n_output))
        if C_discrete is None:
            C_discrete = np.zeros((n_output, n_state))
        if D_discrete is None:
            D_discrete = np.zeros((n_output, n_ctrl))

        self.time_step = time_step
        self.A = A_discrete
        self.B = B_discrete
        self.C = C_discrete
        self.D = D_discrete

    def next_state(self, k, curr_state, curr_ctrl):
        """See base class."""
        del k  # time invariant
        return np.dot(self.A, curr_state) + np.dot(self.B, curr_ctrl)

    def output(self, k, curr_state, curr_ctrl):
        """See base class."""
        del k  # time invariant
        return np.dot(self.C, curr_state) + np.dot(self.D, curr_ctrl)
