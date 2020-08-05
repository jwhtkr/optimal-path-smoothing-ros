"""Contains classes that represent Linear, Time-Invariant systems."""


import scipy.linalg
import numpy as np

from optimal_control.dynamics import dynamics


def _infer_n_state(a_mat, b_mat, c_mat):
    """
    Infer the number of states from `A`, `B`, or `C`.

    If any are None (or evaluate to False in a boolean context), one of the
    others is used. They are not checked for consistency as this should
    arise in other contexts. If all are None (or False), then an exception
    is raised.

    Parameters
    ----------
    a_mat : numpy.ndarray or None
        The A matrix from which to infer the number of states, `n_state`.
    b_mat : numpy.ndarray or None
        The B matrix from which to infer the number of states, `n_state`.
    c_mat : numpy.ndarray or None
        The C matrix from which to infer the number of states, `n_state`.

    Returns
    -------
    int
        The number of states of the system.
    """
    if a_mat is not None:
        return a_mat.shape[0]
    if b_mat is not None:
        return b_mat.shape[0]
    if c_mat is not None:
        return c_mat.shape[1]
    raise TypeError("One of A, B, or C must be valid (not None/empty/etc.).")


def _infer_n_ctrl(b_mat, d_mat):
    """
    Infer the number of control inputs from `B` or `D`.

    If any are None (or evaluate to False in a boolean context), one of the
    others is used. They are not checked for consistency as this should
    arise in other contexts. If all are None (or False), then an exception
    is raised.

    Parameters
    ----------
    b_mat : numpy.ndarray or None
        The B matrix from which to infer the number of control inputs, `n_ctrl`.
    d_mat : numpy.ndarray or None
        The D matrix from which to infer the number of control inputs `n_ctrl`.

    Returns
    -------
    int
        The number of states of the system.
    """
    if b_mat is not None:
        return b_mat.shape[1]
    if d_mat is not None:
        return d_mat.shape[1]
    raise TypeError("One of A, B, or C must be valid (not None/empty/etc.).")


def _infer_n_output(c_mat, d_mat):
    """
    Infer the number of states from `A`, `B`, or `C`.

    If any are None (or evaluate to False in a boolean context), one of the
    others is used. They are not checked for consistency as this should
    arise in other contexts. If all are None (or False), then an exception
    is raised.

    Parameters
    ----------
    c_mat : numpy.ndarray or None
        The C matrix from which to infer the number of outputs, `n_output`.
    d_mat : numpy.ndarray or None
        The D matrix from which to infer the number of outputs, `n_output`.

    Returns
    -------
    int
        The number of states of the system.
    """
    if c_mat is not None:
        return c_mat.shape[0]
    if d_mat is not None:
        return d_mat.shape[0]
    raise TypeError("One of A, B, or C must be valid (not None/empty/etc.).")


class CltiDynamics(dynamics.ContinuousDynamics):
    """
    Represent Continuous Linear Time-Inveriant (CLTI) dynamics.

    Attributes
    ----------
    a_mat : numpy.ndarray
        The A matrix from x_dot = Ax + Bu.
    b_mat : numpy.ndarray
        The B matrix from x_dot = Ax + Bu.
    c_mat : numpy.ndarray
        The C matrix from y = Cx + Du.
    d_mat : numpy.ndarray
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
    a_mat : numpy.ndarray
        The A matrix from x_dot = Ax + Bu.
    b_mat : numpy.ndarray, optional
        The B matrix from x_dot = Ax + Bu. If omitted, a zero matrix is assumed.
        Cannot be omitted if D is omitted.
    c_mat : numpy.ndarray, optional
        The C matrix from y = Cx + Du. If omitted, a zero matrix is assumed.
        Cannot be omitted if D is omitted.
    d_mat : numpy.ndarray, optional
        The D matrix from y = Cx + Du. If omitted, a zero matrix is assumed.
        Cannot be omitted if B or C is omitted.

    """

    def __init__(self, a_mat=None, b_mat=None, c_mat=None, d_mat=None): # noqa: D107
        n_state = _infer_n_state(a_mat, b_mat, c_mat)
        n_ctrl = _infer_n_ctrl(b_mat, d_mat)
        n_output = _infer_n_output(c_mat, d_mat)

        if a_mat is None:
            a_mat = np.zeros((n_state, n_state))
        if b_mat is None:
            b_mat = np.zeros((n_state, n_ctrl))
        if c_mat is None:
            c_mat = np.zeros((n_output, n_state))
        if d_mat is None:
            d_mat = np.zeros((n_output, n_ctrl))

        super(CltiDynamics, self).__init__(n_state, n_ctrl, n_output, {})
        self.a_mat, self.b_mat, self.c_mat, self.d_mat = a_mat, b_mat, c_mat, d_mat

    def state_derivative(self, t, state, ctrl):
        """See base class."""
        del t  # time invariant
        return np.dot(self.a_mat, state) + np.dot(self.b_mat, ctrl)

    def output(self, t, state, ctrl):
        """See base class."""
        del t  # time invariant
        return np.dot(self.c_mat, state)  + np.dot(self.d_mat, ctrl)

    def linearize(self, state_star, ctrl_star):
        """See base class."""
        return self.a_mat, self.b_mat, self.c_mat, self.d_mat

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
        tmp = np.concatenate((self.a_mat, self.b_mat), axis=1)
        tmp = np.concatenate((tmp, np.zeros((self.n_ctrl,
                                             self.n_state+self.n_ctrl))))
        tmp = scipy.linalg.expm(tmp*time_step)  # pylint: disable=no-member
        a_bar = tmp[0:self.n_state, 0:self.n_state]
        b_bar = tmp[0:self.n_state, self.n_state:self.n_state+self.n_ctrl]

        return DltiDynamics(time_step, a_bar, b_bar, self.c_mat, self.d_mat)


class DltiDynamics(dynamics.DiscreteDynamics):
    """
    Represent Discrete Linear Time-Invariant (DLTI) dynamics.

    Attributes
    ----------
    a_mat : numpy.ndarray
        The A matrix of the discrete dynamic equation x_{k+1} = Ax_k + Bu_k.
    b_mat : numpy.ndarray
        The B matrix of the discrete dynamic equation x_{k+1} = Ax_k + Bu_k.
    c_mat : numpy.ndarray
        The C matrix of the discrete dynamic equation y_k = Cx_k + Du_k.
    d_mat : numpy.ndarray
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
    a_discrete : numpy.ndarray
        The A matrix of the discrete dynamic equation x_{k+1} = Ax_k + Bu_k.
    b_discrete : numpy.ndarray, optional
        The B matrix of the discrete dynamic equation x_{k+1} = Ax_k + Bu_k. If
        omitted it is assumed to be a zero matrix.
    c_discrete : numpy.ndarray, optional
        The C matrix of the discrete dynamic equation y_k = Cx_k + Du_k. If
        omitted it is assumed to be a zero matrix.
    d_discrete : numpy.ndarray, optional
        The D matrix of the discrete dynamic equation y_k = Cx_k + Du_k. If
        omitted it is assumed to be a zero matrix.

    """

    def __init__(self, time_step, a_discrete, b_discrete=None,
                 c_discrete=None, d_discrete=None):
        """Init."""
        n_state = _infer_n_state(a_discrete, b_discrete, c_discrete)
        n_ctrl = _infer_n_ctrl(b_discrete, d_discrete)
        n_output = _infer_n_output(c_discrete, d_discrete)

        super(DltiDynamics, self).__init__(n_state, n_ctrl, n_output, {})

        if b_discrete is None:
            b_discrete = np.zeros((n_state, n_output))
        if c_discrete is None:
            c_discrete = np.zeros((n_output, n_state))
        if d_discrete is None:
            d_discrete = np.zeros((n_output, n_ctrl))

        self.time_step = time_step
        self.a_mat = a_discrete
        self.b_mat = b_discrete
        self.c_mat = c_discrete
        self.d_mat = d_discrete

    def next_state(self, k, curr_state, curr_ctrl):
        """See base class."""
        del k  # time invariant
        return np.dot(self.a_mat, curr_state) + np.dot(self.b_mat, curr_ctrl)

    def output(self, k, curr_state, curr_ctrl):
        """See base class."""
        del k  # time invariant
        return np.dot(self.c_mat, curr_state) + np.dot(self.d_mat, curr_ctrl)
