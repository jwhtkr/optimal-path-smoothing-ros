"""Contains the class for performing optimal trajectory smoothing."""


import scipy.sparse
import numpy as np

import optimal_control.opt_ctrl.direct.fixed_time as fixed_time
import optimal_control.dynamics.integrator_dynamics as int_dyn
import optimal_control.objectives.quadratic_cost as quad_cost
import optimal_control.constraints.linear_constraints as lin_const
import optimal_control.solvers.osqp_solver as osqp


class SmoothPathLinear(fixed_time.FixedTime):
    """
    Represents a linear trajectory smoothing optimal control problem.

    Attributes
    ----------
    n_smooth : int
        The "level" of smoothness, or the class of continuity desired. Also, the
        number of times the input is integrated to become the trajectory.
    n_dim : int
        The dimensionality of the path (i.e., 2D or 3D).
    initial_state : numpy.ndarray
        The state at the initial time. It must be passed in as an array of shape
        (n_dim, n_smooth)
    See base class for additional attributes.

    Parameters
    ----------
    See base class for additional parameters.
    n_step : int
        The number of time steps in the path.
    initial_state : numpy.ndarray, optional
        The state at the initial time. If unspecified the first state of the
        desired path is used.


    """
    def __init__(self, constraints, cost, solver, n_step, initial_state,
                 t_final=0., time_step=0., **kwargs):
        self.n_dim, self.n_smooth = initial_state.shape
        self.initial_state = initial_state.reshape(-1, 1, order="F")
        dynamics = self._integrator_dynamics()

        super(SmoothPathLinear, self).__init__(dynamics, constraints, cost,
                                                  solver, n_step, t_final,
                                                  time_step, **kwargs)

    def update(self, **kwargs):
        """
        Discretize and aggregate the problem based on the current attributes.

        The dynamics are to be discretized (if not already), and the states and
        inputs aggregated. This is done based on the current attributes, and
        must be called if any of them change. It can be an expensive operation
        so only call it when necessary.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the `setup` function of the
            solver.

        """
        dyn_constraint = self._dynamic_constraint()

        self.direct_cost = self._aggregate_cost()
        self.direct_constraints = self._aggregate_constraints(dyn_constraint)

        super(SmoothPathLinear, self).update(**kwargs)

    def solve(self, warm_start=None, **kwargs):
        y = super(SmoothPathLinear, self).solve(warm_start, **kwargs)
        return (y[:self.n_dim*self.n_smooth*self.n_step],
                y[self.n_dim*self.n_smooth*self.n_step:])

    def _integrator_dynamics(self):
        """
        Create the integrator dynamics.

        Returns
        -------
        int_dyn.ContinuousIntegratorDynamics
            Integrator dynamics for use in smoothing a trajectory.

        """
        return int_dyn.ContinuousIntegratorDynamics(self.n_dim, self.n_smooth)

    def _dynamic_constraint(self):
        """
        Discretize and convert the dynamics to a linear, aggregated constraint.

        Returns
        -------
        lin_const.LinearEqualityConstraint
            The dynamics represented as a linear equality constraint.

        """
        discrete_dyn = self.dynamics.discretize(self.time_step)

        M = discrete_dyn.n_state*self.n_step

        A_x = -scipy.sparse.eye(m=M, n=discrete_dyn.n_state*self.n_step,
                                format="dok")
        A_u = scipy.sparse.dok_matrix((M, discrete_dyn.n_ctrl*(self.n_step-1)))
        b = np.zeros((M,))

        b[:discrete_dyn.n_state] = -self.initial_state.flatten()

        for i in range(1, self.n_step):
            row_begin = i * discrete_dyn.n_state
            row_end = row_begin + discrete_dyn.n_state
            x_col_begin = (i-1) * discrete_dyn.n_state
            x_col_end = x_col_begin + discrete_dyn.n_state
            u_col_begin = (i-1) * discrete_dyn.n_ctrl
            u_col_end = u_col_begin + discrete_dyn.n_ctrl

            A_x[row_begin:row_end, x_col_begin:x_col_end] = discrete_dyn.A
            A_u[row_begin:row_end, u_col_begin:u_col_end] = discrete_dyn.B

        b = b.reshape(-1, 1)

        def dyn_const(t):
            """Return the matrices of the dynamic constraint."""
            del t
            return A_x, A_u

        def dyn_val(t):
            """Return the vector of the dynamic constraint."""
            del t
            return b

        return lin_const.LinearEqualityConstraint(dyn_const, dyn_val)

    def _aggregate_cost(self):
        """
        Aggregate the cost function to be a function of the entire path.

        Returns
        -------
        quad_cost.ContinuousQuadraticCost
            The aggregated cost.

        """
        n_state = self.n_dim * self.n_smooth
        M_x = n_state*self.n_step

        Q_sparse = scipy.sparse.csc_matrix(self.cost.inst_state_cost)
        Q_sub = scipy.sparse.block_diag([Q_sparse
                                         for _ in range(self.n_step-1)],
                                        format="csc")
        Q_list = [[Q_sub, None],
                  [None, scipy.sparse.csc_matrix((n_state, n_state))]]
        Q = scipy.sparse.bmat(Q_list, format="csc")

        R = scipy.sparse.block_diag([self.cost.inst_ctrl_cost
                                     for _ in range(self.n_step-1)],
                                    format="csc")

        S_list = [[scipy.sparse.csc_matrix((M_x-n_state, M_x-n_state)), None],
                  [None, self.cost.term_state_cost]]
        S = scipy.sparse.bmat(S_list, format="csc")

        x_d = np.concatenate([self.cost.desired_state(i*self.time_step)
                              for i in range(self.n_step)]).reshape(-1,1)
        u_d = np.concatenate([self.cost.desired_ctrl(i*self.time_step)
                              for i in range(self.n_step-1)]).reshape(-1,1)

        def desired_state(t):
            """Return the desired state."""
            del t
            return x_d

        def desired_ctrl(t):
            """Return the desired input control."""
            del t
            return u_d

        return quad_cost.ContinuousQuadraticCost(Q, R, S, desired_state,
                                                 desired_ctrl)

    def _aggregate_constraints(self, dyn_constraint):
        """
        Aggregate the constraints to be a function of the entire path.


        Parameters
        ----------
        dyn_constraint : lin_const.LinearEqualityConstraint
            The dynamics represented as an aggregated constraint to be added to
            the other constraints once they are aggregated as well.

        Returns
        -------
        lin_const.LinearConstraints
            The aggregated linear constraints.

        """
        if self.constraints is None:
            return lin_const.LinearConstraints([dyn_constraint.eq_mats],
                                               [dyn_constraint.eq_val])
        eq_const = [[], [], []]

        ineq_const = [[], [], []]

        for i in range(self.n_step):
            t = self.time_step*i

            eq = self.constraints.equality_mat_vec(t)
            ineq = self.constraints.inequality_mat_vec(t)

            eq_const[0].append(eq[0])
            eq_const[1].append(eq[1])
            eq_const[2].append(eq[2])

            ineq_const[0].append(ineq[0])
            ineq_const[1].append(ineq[1])
            ineq_const[2].append(ineq[2])

        eq_const = (scipy.sparse.block_diag(eq_const[0]),
                    scipy.sparse.block_diag(eq_const[1]),
                    np.concatenate(eq_const[2]))

        ineq_const = (scipy.sparse.block_diag(ineq_const[0]),
                      scipy.sparse.block_diag(ineq_const[1]),
                      np.concatenate(ineq_const[2]))

        def eq_mats(t):
            """Return the matrices of the aggregate equality constraint."""
            del t
            return eq_const[0], eq_const[1]

        def eq_val(t):
            """Return the vector of the aggregate equality constraint."""
            del t
            return eq_const[2]

        def ineq_mats(t):
            """Return the matrices of the aggregate inequality constraint."""
            del t
            return ineq_const[0], ineq_const[1]

        def ineq_bound(t):
            """Return the vector of the aggregate inequality constraint."""
            del t
            return ineq_const[2]

        return lin_const.LinearConstraints([eq_mats, dyn_constraint.eq_mats],
                                           [eq_val, dyn_constraint.eq_val],
                                           [ineq_mats],
                                           [ineq_bound])


if __name__ == "__main__":
    # pylint: disable=invalid-name
    pts = [(0., -1., 1.),
           (10., 2., 2.),
           (20., 1., 0.),
           (30., -2., -1.),
           (40., -1., 1.)]

    def path(t_in):
        """Return the path based on the pts in function scope."""
        if t_in < 0:
            raise ValueError("t must be greater than 0.")

        def interpolate(arg1, arg1_prev, arg1_next, arg2_prev, arg2_next):
            """Interpolate b based on a (linearly)."""
            return (arg1-arg1_prev)/(arg1_next-arg1_prev)*(arg2_next-arg2_prev)\
                + arg2_prev

        for i, pt in enumerate(pts):
            t, x, y = pt
            if pt is pts[-1]:
                return np.array([x, y, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
            t_next, x_next, y_next = pts[i+1]

            if t_in >= t and t_in < t_next:
                x_interp = interpolate(t_in, t, t_next, x, x_next)
                y_interp = interpolate(t_in, t, t_next, y, y_next)
                x_vel = (x_next-x)/(t_next-t)
                y_vel = (y_next-y)/(t_next-t)
                vec = [x_interp, y_interp, x_vel, y_vel, 0, 0, 0, 0]

                return np.array(vec).reshape(-1, 1)

    time = np.arange(40, step=0.5)
    x_desired = np.zeros((8, len(time)))

    for k, time_inst in enumerate(time):
        x_desired[:, k] = path(time_inst).flatten()

    import matplotlib.pyplot as plt

    Q_mat = np.diag([1., 1., 0., 0., 1., 1., 1., 1.])
    R_mat = np.diag([10., 10.])
    S_mat = np.diag([10., 10., 0., 0., 0., 0., 0., 0.])
    cst = quad_cost.ContinuousQuadraticCost(Q_mat, R_mat, S_mat,
                                            desired_state=path)
    cnstrnts = None
    slvr = osqp.OSQP()

    x_0 = x_desired[:, 0].reshape(2, 4, order="F").copy()
    x_0[1, 0] = 0.5
    x_0[:,1:] = 0.
    opt_ctrl = SmoothPathLinear(cnstrnts, cst, slvr, len(time), x_0,
                                   t_final=time[-1])

    st, ctl = opt_ctrl.solve()
    st = st.reshape(8, -1, order="F")
    ctl = ctl.reshape(2, -1, order="F")

    plt.plot(x_desired[0, :], x_desired[1, :], st[0, :], st[1, :])
    fig, axs = plt.subplots(8, 1)
    for k, ax in enumerate(axs):
        ax.plot(time, x_desired[k, :], time, st[k, :])
    fig, axs = plt.subplots(2, 1)
    for k, ax in enumerate(axs):
        ax.plot(time[:-1], ctl[k, :])
    plt.show()
    # pylint: enable=invalid-name
