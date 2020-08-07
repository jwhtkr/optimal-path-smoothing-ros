"""Contains the class for performing optimal trajectory smoothing."""


import numpy as np

import optimal_control.sparse_utils as sparse
import optimal_control.opt_ctrl.direct.fixed_time as fixed_time
import optimal_control.dynamics.integrator_dynamics as int_dyn
import optimal_control.objectives.quadratic_cost as quad_cost
import optimal_control.constraints.linear_constraints as lin_const

# pylint: disable=unused-import
import optimal_control.solvers.osqp_solver as osqp
import optimal_control.solvers.gurobi_solver as gurobi
# pylint: enable=unused-import


def interpolate(arg1, arg1_prev, arg1_next, arg2_prev, arg2_next):
    """Interpolate arg2 based on arg1 (linearly)."""
    return (arg1-arg1_prev)/(arg1_next-arg1_prev)*(arg2_next-arg2_prev)\
        + arg2_prev


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
                 t_final=0., time_step=0., **kwargs): # noqa: D107
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
        """See base class."""
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

        m_mat = discrete_dyn.n_state*self.n_step

        a_x = -sparse.eye(m=m_mat, n=discrete_dyn.n_state*self.n_step, format="dok")
        a_u = sparse.dok_matrix((m_mat, discrete_dyn.n_ctrl*(self.n_step-1)))
        b_mat = np.zeros((m_mat,))

        b_mat[:discrete_dyn.n_state] = -self.initial_state.flatten()

        for i in range(1, self.n_step):
            row_begin = i * discrete_dyn.n_state
            row_end = row_begin + discrete_dyn.n_state
            x_col_begin = (i-1) * discrete_dyn.n_state
            x_col_end = x_col_begin + discrete_dyn.n_state
            u_col_begin = (i-1) * discrete_dyn.n_ctrl
            u_col_end = u_col_begin + discrete_dyn.n_ctrl

            a_x[row_begin:row_end, x_col_begin:x_col_end] = discrete_dyn.a_mat
            a_u[row_begin:row_end, u_col_begin:u_col_end] = discrete_dyn.b_mat

        b_mat = b_mat.reshape(-1, 1)

        def dyn_const(t):
            """Return the matrices of the dynamic constraint."""
            del t
            return a_x, a_u

        def dyn_val(t):
            """Return the vector of the dynamic constraint."""
            del t
            return b_mat

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
        m_x = n_state*self.n_step

        q_sparse = sparse.coo_matrix(self.cost.inst_state_cost)
        q_sub = sparse.block_diag([q_sparse for _ in range(self.n_step-1)])
        q_list = [[q_sub, None],
                  [None, sparse.coo_matrix((n_state, n_state))]]
        q_mat = sparse.bmat(q_list, format="csc")

        r_mat = sparse.block_diag([self.cost.inst_ctrl_cost
                               for _ in range(self.n_step-1)])

        s_list = [[sparse.coo_matrix((m_x-n_state, m_x-n_state)), None],
                  [None, self.cost.term_state_cost]]
        s_mat = sparse.bmat(s_list, format="csc")

        x_d = np.concatenate([self.cost.desired_state(i*self.time_step)
                              for i in range(self.n_step)]).reshape(-1, 1)
        u_d = np.concatenate([self.cost.desired_ctrl(i*self.time_step)
                              for i in range(self.n_step-1)]).reshape(-1, 1)

        def desired_state(t):
            """Return the desired state."""
            del t
            return x_d

        def desired_ctrl(t):
            """Return the desired input control."""
            del t
            return u_d

        return quad_cost.ContinuousQuadraticCost(q_mat, r_mat, s_mat, desired_state,
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

            eq_mat = self.constraints.equality_mat_vec(t)
            ineq = self.constraints.inequality_mat_vec(t)

            eq_const[0].append(eq_mat[0])
            eq_const[1].append(eq_mat[1])
            eq_const[2].append(eq_mat[2])

            ineq_const[0].append(ineq[0])
            ineq_const[1].append(ineq[1])
            ineq_const[2].append(ineq[2])

        n_ctrl = self.constraints.n_ctrl
        eq_const = (sparse.block_diag(eq_const[0]),
                    sparse.block_diag(eq_const[1]),
                    np.concatenate(eq_const[2]))

        ineq_const = (sparse.block_diag(ineq_const[0]),
                      sparse.block_diag(ineq_const[1]),
                      np.concatenate(ineq_const[2]))

        def eq_mats(t):
            """Return the matrices of the aggregate equality constraint."""
            del t
            return eq_const[0], eq_const[1].tocsc()[:, :-n_ctrl].tocoo()

        def eq_val(t):
            """Return the vector of the aggregate equality constraint."""
            del t
            return eq_const[2]

        def ineq_mats(t):
            """Return the matrices of the aggregate inequality constraint."""
            del t
            return ineq_const[0], ineq_const[1].tocsc()[:, :-n_ctrl].tocoo()

        def ineq_bound(t):
            """Return the vector of the aggregate inequality constraint."""
            del t
            return ineq_const[2]

        return lin_const.LinearConstraints(eq_mat_list=[eq_mats,
                                                        dyn_constraint.eq_mats],
                                           eq_val_list=[eq_val,
                                                        dyn_constraint.eq_val],
                                           ineq_mat_list=[ineq_mats],
                                           ineq_bound_list=[ineq_bound])


if __name__ == "__main__":
    import time as time_module
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

    # def term_eq_mats(t):
    #     """Return terminal constraint matrices."""
    #     if t == time[-1]:
    #         return (np.eye(8), np.zeros((8, 2)))
    #     return (np.zeros((0, 8)), np.zeros((0, 2)))

    # def term_eq_val(t):
    #     """Return terminal constraint values."""
    #     if t == time[-1]:
    #         return x_desired[:, -1].reshape(-1, 1)
    #     return np.zeros((0, 1))
    # cnstrnts = lin_const.LinearConstraints(eq_mat_list=[term_eq_mats],
    #                                        eq_val_list=[term_eq_val])
    eq_matrices = (np.eye(N=2, M=8), np.zeros((2, 2)))
    # term_cnstrnt = lin_const.LinearTimeInstantConstraint(time[-1],
    #                                                      eq_mats,
    #                                                      x_desired[:2, -1].reshape(-1, 1))
    # mid_cnstrnt = lin_const.LinearTimeInstantConstraint(time[len(time)//2],
    #                                                     eq_mats,
    #                                                     x_desired[:2, len(time)//2]
    #                                                         .reshape(-1, 1))
    # inst_cnstrnts = [term_cnstrnt, mid_cnstrnt]

    inst_cnstrnts = []
    for it in range(10, len(time), 10):
        x_des = x_desired[:2, it].reshape(-1, 1)
        inst_cnstrnts.append(lin_const.LinearTimeInstantConstraint(time[it],
                                                                   eq_matrices,
                                                                   x_des))
    cnstrnts = lin_const.LinearConstraints(eq_constraints=inst_cnstrnts)
    # cnstrnts = None
    # slvr = osqp.OSQP()
    slvr = gurobi.Gurobi()

    x_0 = x_desired[:, 0].reshape(2, 4, order="F").copy()
    x_0[1, 0] = 0.5
    x_0[:, 1:] = 0.
    t_start = time_module.time()
    opt_ctrl = SmoothPathLinear(cnstrnts, cst, slvr, len(time), x_0,
                                t_final=time[-1])

    st, ctl = opt_ctrl.solve()
    print("Solve Time: {}".format(time_module.time() - t_start))
    st = st.reshape(8, -1, order="F")
    ctl = ctl.reshape(2, -1, order="F")

    plt.plot(x_desired[0, :], x_desired[1, :], st[0, :], st[1, :])
    labels1 = [r"$x$", r"$y$", r"$v_x$", r"$v_y$", r"$a_x$", r"$a_y$", r"$x^{(3)}$", r"$y^{(3)}$"]
    labels2 = [r"$u_x$", r"$u_y$"]
    fig, axs = plt.subplots(8, 1)
    for k, ax in enumerate(axs):
        ax.plot(time, x_desired[k, :], time, st[k, :])
        ax.set_ylabel(labels1[k])
    fig, axs = plt.subplots(2, 1)
    for k, ax in enumerate(axs):
        ax.plot(time[:-1], ctl[k, :])
        ax.set_ylabel(labels2[k])
    plt.show()
    # pylint: enable=invalid-name
