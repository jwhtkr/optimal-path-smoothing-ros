"""Contains the classes to represent integrator dynamics."""

import numpy as np

from optimal_control.dynamics import lti_dynamics

class ContinuousIntegratorDynamics(lti_dynamics.CltiDynamics):
    """
    Represent continuous integrator dynamics.

    Represents the dynamics of a chain of integrators with arbitrary
    dimensionality. In other words, there is an input of size n (`n_dim`) that
    is integrated m (`n_int`) times to output a position in n-dimensional space.
    This is a linear, time-invariant system.

    Attributes
    ----------
    a_mat : numpy.ndarray
        The A matrix from x_dot = Ax + Bu. It is of shape
        (`n_dim`*`n_int`, `n_dim`*`n_int`).
    b_mat : numpy.ndarray
        The B matrix from x_dot = Ax + Bu. It is of shape
        (`n_dim`*`n_int`, `n_dim`)
    c_mat : numpy.ndarray
        The C matrix from y = Cx + Du. It is of shape
        (`n_dim`, `n_dim`*`n_int`).
    d_mat : numpy.ndarray
        The D matrix from y = Cx + Du. It is of shape (`n_dim`, `n_dim`).
    n_dim : int
        The dimensionality of the problem (i.e., 2->plane/2D, 3->3D, etc.).
    n_int : int
        The number of integrators between the input and the output.

    Parameters
    ----------
    n_dim : int
        The dimensionality of the problem (i.e., 2->plane/2D, 3->3D, etc.).
    n_int : int
        The number of integrators between the input and the output.
    """

    def __init__(self, n_dim, n_int): # noqa: D107
        self.n_dim = n_dim
        self.n_int = n_int
        a_mat, b_mat, c_mat, d_mat = self._integrator_dynamics()
        super(ContinuousIntegratorDynamics, self).__init__(a_mat, b_mat, c_mat, d_mat)

    def _integrator_dynamics(self):
        """
        Create the A, B, C, and D matrices for a chain of integrators.

        Creates the A, B, C, and D matrices in controller canonical form for a
        chain of integrators.

        Returns
        -------
        a_mat : numpy.ndarray
            The A matrix for a chain of integrators.
        b_mat : numpy.ndarray
            The B matrix for a chain of integrators.
        c_mat : numpy.ndarray
            The C matrix for a chain of integrators.
        d_mat : numpy.ndarray
            The D matrix for a chain of integrators.
        """
        a_mat = np.eye(self.n_dim*self.n_int, k=self.n_dim)
        b_mat = np.eye(self.n_dim*self.n_int,
                   M=self.n_dim,
                   k=-self.n_dim*(self.n_int-1))
        c_mat = np.eye(self.n_dim, M=self.n_dim*self.n_int)
        d_mat = np.zeros((self.n_dim, self.n_dim))

        return a_mat, b_mat, c_mat, d_mat

    def discretize(self, time_step, method="exact"):
        """See base class."""
        return super(ContinuousIntegratorDynamics, self).discretize(time_step,
                                                                    method)
