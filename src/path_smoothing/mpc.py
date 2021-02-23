"""Do MPC with the path smoothing setup."""

import numpy as np
import scipy.sparse as sparse
import gurobipy as gp

import path_smoothing.smooth_path_lp as smooth


def dyn_constraints(ndim, nint, nstep, time_step):
    """Create the constraint for the discrete, differentially flat dynamics."""
    initial_state = np.zeros((ndim, nint))
    a_mat, b_vec_0 = smooth.discrete_dynamic_equalities(ndim, nint, nstep,
                                                initial_state, time_step)
    def b_vec(t, x, u, xd):  # pylint: disable=invalid-name
        del t, u, xd
        b_vec_0[:ndim*nint] = -x.flatten(order="F")
        return b_vec_0

    return (a_mat.tocsr(), False), (b_vec, True), "eq"


class DiffFlatMpc(object):
    """
    Follows a desired trajectory with MPC in differentially flat space.

    Attributes
    ----------
    model : gurobipy.Model
        The Gurobi model for the optimization.
    ndim : int
        The number of dimensions (i.e. 2D vs 3D, etc.).
    nint : int
        The number of derivatives in the state. One more derivative is the
        input.
    nstep : int
        The number of steps in the horizon to optimize over.
    time_step : float
        The time step in between each point in the trajectory. This is the time
        step used for discretization as well.
    objective : tuple of ((scipy.sparse or func, bool), (scipy.sparse or func, bool))
        A tuple that defines the quadratic cost as ((P_x, bool), (P_u, bool)). A
        true boolean value indicates that the corresponding cost term is
        callable with the signature (t, x, u, xd) -> P where P is a positive
        semi-definite matrix in sparse format. If false, the element is a
        positive semi-definite matrix in sparse format. The overall cost is
        defined as (x-xd)@P_x@(x-xd) + u@P_u@u
    constraints : list of tuples of ((scipy.sparse or func, bool),
                                     (numpy.ndarray or func, bool),
                                     str)
        A list of linear constraints described as the tuple of
        ((A, bool), (b, bool), string) where a true value for the two booleans
        indicates that A or b, respectively, are callable with signature
        (t, x, u, xd) -> A or b, where t is the current initial time of the MPC
        iteration, x is the current state, u is the current input, and xd is the
        current desired trajectory for the MPC horizon. A false value indicates
        that the respective element is just the array (i.e., it does not
        change). The string is one of ["ineq", "eq"] indicating that
        A@[x; u] <= b or A@[x; u] == b, respectively.
    state : gurobipy.MVar
        The state vector of the optimization.
    control : gurobipy.MVar
        The control vector of the optimization.


    Parameters
    ----------
    ndim : int
        The number of dimensions (i.e. 2D vs 3D, etc.).
    nint : int
        The number of derivatives in the state. One more derivative is the
        input.
    nstep : int
        The number of steps in the horizon to optimize over.
    time_step : float
        The time step in between each point in the trajectory. This is the time
        step used for discretization as well.
    objective : tuple of ((scipy.sparse or func, bool), (scipy.sparse or func, bool))
        A tuple that defines the quadratic cost as ((P_x, bool), (P_u, bool)). A
        true boolean value indicates that the corresponding cost term is
        callable with the signature (t, x, u, xd) -> P where P is a positive
        semi-definite matrix in sparse format. If false, the element is a
        positive semi-definite matrix in sparse format. The overall cost is
        defined as (x-xd)@P_x@(x-xd) + u@P_u@u
    constraints : list of tuples of ((scipy.sparse or func, bool),
                                     (numpy.ndarray or func, bool),
                                     str)
        A list of linear constraints described as the tuple of
        ((A, bool), (b, bool), string) where a true value for the two booleans
        indicates that A or b, respectively, are callable with signature
        (t, x, u, xd) -> A or b, where t is the current initial time of the MPC
        iteration, x is the current state, u is the current input, and xd is the
        current desired trajectory for the MPC horizon. A false value indicates
        that the respective element is just the array (i.e., it does not
        change). The string is one of ["ineq", "eq"] indicating that
        A@[x; u] <= b or A@[x; u] == b, respectively.
    """

    def __init__(self, ndim, nint, nstep, time_step, objective, constraints=()):  # noqa: D107
        self.model = gp.Model()
        self.ndim, self.nint, self.nstep = ndim, nint, nstep
        self.time_step = time_step
        self.constraints = list(constraints)
        self.constraints.append(dyn_constraints(ndim, nint, nstep, time_step))
        self._constraints = []
        self.objective = objective
        self._objective = None
        self.state = None
        self.control = None
        self.param_vec = None

        self._init_model()

    def _init_model(self):
        self.state = self.model.addMVar((self.ndim*self.nint*self.nstep,),
                                        lb=-gp.GRB.INFINITY,
                                        vtype=gp.GRB.CONTINUOUS,
                                        name="state")
        self.control = self.model.addMVar((self.ndim*(self.nstep-1),),
                                          lb=-gp.GRB.INFINITY,
                                          vtype=gp.GRB.CONTINUOUS,
                                          name="control")
        self._update_model(0.,
                           np.zeros((self.ndim, self.nint)),
                           np.zeros((self.ndim,)),
                           np.zeros((self.ndim, self.nint, self.nstep)))
        self.model.setParam("TimeLimit", self.time_step)
        self.model.setParam("outputFlag", 0)

    def _update_model(self, time, curr_state, curr_control, des_traj):
        if self._constraints:
            self._update_constraints(time, curr_state, curr_control, des_traj)
        else:
            self._init_constraints(time, curr_state, curr_control, des_traj)
        if self._objective:
            self._update_objective(time, curr_state, curr_control, des_traj)
        else:
            self._init_objective(time, curr_state, curr_control, des_traj)
        # self._update_constraints(time, curr_state, curr_control, des_traj)
        # self._update_objective(time, curr_state, curr_control, des_traj)

    def _init_constraints(self, time, curr_state, curr_control, des_traj):
        for i, constraint in enumerate(self.constraints):
            expr = self._constraint_expr(constraint, time, curr_state, curr_control, des_traj)
            self._constraints.append(self.model.addConstr(expr, name=f"c_{i}"))

    def _constraint_expr(self, constraint, time, curr_state, curr_control, des_traj):
        a_mat, b_vec = constraint[0], constraint[1]
        if a_mat[1]:
            a_mat = a_mat[0](time, curr_state, curr_control, des_traj)
        else:
            a_mat = a_mat[0]
        if b_vec[1]:
            b_vec = b_vec[0](time, curr_state, curr_control, des_traj)
        else:
            b_vec = b_vec[0]
        a_mat_x = a_mat[:, :self.ndim*self.nint*self.nstep]
        a_mat_u = a_mat[:, self.ndim*self.nint*self.nstep:]
        if constraint[2] == "eq":
            expr = a_mat_x @ self.state + a_mat_u @ self.control == b_vec
        elif constraint[2] == "ineq":
            expr = a_mat_x @ self.state + a_mat_u @ self.control <= b_vec
        else:
            raise ValueError(f"Constraint {i} was given sense "
                                "{constraint[2]}, which is  not one of "
                                "[\"eq\", \"ineq\"]")
        return expr

    def _init_objective(self, time, curr_state, curr_control, des_traj):
        p_x, p_u = self.objective[0], self.objective[1]
        if p_x[1]:
            p_x = p_x[0](time, curr_state, curr_control, des_traj)
        else:
            p_x = p_x[0]
        if p_u[1]:
            p_u = p_u[0](time, curr_state, curr_control, des_traj)
        else:
            p_u = p_u[0]
        x_des = des_traj.flatten(order="F")
        quad_x = self.state @ p_x @ self.state
        quad_u = self.control @ p_u @ self.control
        lin_x = -2*(x_des @ p_x) @ self.state
        self._objective = self.model.setObjective(quad_x + quad_u + lin_x)

    def _update_constraints(self, time, curr_state, curr_control, des_traj):
        for i, constraint in enumerate(self.constraints):
            if constraint[1]:
                self.model.remove(self._constraints[i])
                expr = self._constraint_expr(constraint, time, curr_state, curr_control, des_traj)
                self._constraints[i] = self.model.addConstr(expr, name=f"c_{i}")
            # TODO: Update the constraints that change instead of replace them

    def _update_objective(self, time, curr_state, curr_control, des_traj):
        # TODO: only update when/what is changed in the objective.
        if self.objective[0][1] or self.objective[1][1]:
            self._init_objective(time, curr_state, curr_control, des_traj)

    def step(self, time, curr_state, curr_control, des_traj):
        """Generate the next step of MPC. Return the horizon's trajectory."""
        self._update_model(time, curr_state, curr_control, des_traj)
        self.model.optimize()
        state = self.state.x.reshape((self.ndim, self.nint, self.nstep),
                                     order="F")
        control = self.control.x.reshape((self.ndim, self.nstep-1), order="F")
        return state, control


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    n = 200
    upper_lim = n/2
    lower_lim = 0
    xd = np.concatenate([np.linspace(lower_lim, upper_lim, num=n//2),
                         np.full((n//2,), upper_lim)])
    yd = np.concatenate([np.full((n//2,), lower_lim),
                         np.linspace(lower_lim, upper_lim, num=n//2)])
    xd_vec = np.concatenate([xd.reshape(1, -1), yd.reshape(1, -1),
                             np.zeros((6, n))],
                            axis=0)
    xd_mat = xd_vec.reshape((2, 4, -1), order='F')
    xd_mat[:, 1, :50] = np.array([1, 0]).reshape(-1, 1)
    xd_mat[:, 1, -50:] = np.array([0, 1]).reshape(-1, 1)

    ndim, nint, nstep = xd_mat.shape
    nhorizon = 50
    q_mat = np.diag([1, 1, 0, 0, 10, 10, 10, 10])
    r_mat = np.diag([1, 1])
    s_mat = q_mat*10
    # a_mat = np.empty((0, ndim, nint, nstep))
    a_mat = np.array([[1, 0], [0, -1], [lower_lim - upper_lim, upper_lim - lower_lim]])
    a_mat = sparse.bmat([[a_mat, sparse.coo_matrix((a_mat.shape[0], ndim*(nint-1)))]])
    a_mat_full = np.concatenate([a_mat.toarray(), np.zeros((a_mat.shape[0], ndim))], axis=1)
    a_mat_full = a_mat_full.reshape((3, ndim, nint+1), order="F")
    a_mat_full = np.concatenate([a_mat_full[:,:,:,np.newaxis] for _ in range(nstep)], axis=3)
    a_mat = sparse.block_diag([a_mat for _ in range(nhorizon)])
    a_mat = sparse.bmat([[a_mat, sparse.coo_matrix((a_mat.shape[0], ndim*(nhorizon-1)))]])
    a_mat = a_mat.tocsc()
    # b_mat = np.empty((0, nstep))
    b_mat = np.array([upper_lim, -lower_lim, 0])
    b_mat_full = np.concatenate([b_mat for _ in range(nstep)])
    b_mat = np.concatenate([b_mat for _ in range(nhorizon)])
    dt = 0.2

    # P_x, P_u, q_x, q_u = smooth.quadratic_cost(xd_mat[:, :, :nhorizon], q_mat,
    #                                            r_mat, s_mat, ndim, nint,
    #                                            nhorizon)
    def P_x(t, x, u, xd):
        del t, x, u
        ret_val, _, _, _ = smooth.quadratic_cost(xd, q_mat, r_mat, s_mat, ndim,
                                                 nint, nhorizon)
        return ret_val

    def P_u(t, x, u, xd):
        del t, x, u
        _, ret_val, _, _ = smooth.quadratic_cost(xd, q_mat, r_mat, s_mat, ndim,
                                                 nint, nhorizon)
        return ret_val

    constraint = ((a_mat, False), (b_mat, False), "ineq")

    mpc = DiffFlatMpc(ndim, nint, nhorizon, dt, ((P_x, True), (P_u, True)),
                      constraints=[constraint])
    opt = smooth.smooth_path_qp(xd_mat, q_mat, r_mat, s_mat,
                                a_mat_full,
                                b_mat_full.reshape((3, nstep), order="F"),
                                (), dt)

    # fig, ax = plt.subplots()
    plt.plot(xd, yd)

    solve_times = []
    curr_x = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.]])
    curr_u = np.zeros((ndim,))
    result_traj = [(curr_x, curr_u)]
    mpc_line = None
    for i in range(nstep-nhorizon):
        curr_xd = xd_mat[:, :, i:i+nhorizon]
        tic = time.time()
        traj, ctrl = mpc.step(i, curr_x, curr_u, curr_xd)
        solve_times.append(time.time()-tic)
        print(f"Time: {solve_times[-1]:.3f}")
        # if not mpc_line:
        #     mpc_line, = plt.plot(traj[0, 0, :], traj[1, 0, :])
        #     pos_line, = plt.plot(curr_x[0, 0], curr_x[1, 0], 'x')
        #     pass
        # else:
        #     mpc_line.set_xdata(traj[0, 0, :])
        #     mpc_line.set_ydata(traj[1, 0, :])
        #     pos_line.set_xdata(curr_x[0, 0])
        #     pos_line.set_ydata(curr_x[1, 0])
        #     pass

        curr_x = traj[:, :, 1]
        curr_u = ctrl[:, 0]
        result_traj.append((curr_x, curr_u))
        # plt.pause(0.05)

    x = np.array([tmp[0][0, 0] for tmp in result_traj])
    y = np.array([tmp[0][1, 0] for tmp in result_traj])
    plt.plot(x, y)
    opt_traj = opt[0].reshape((ndim, nint, nstep), order="F")
    x_opt, y_opt = opt_traj[0, 0, :], opt_traj[1, 0, :]
    plt.plot(x_opt, y_opt)
    print(f"Avg. Solve Time: {sum(solve_times)/len(solve_times):.3f}")
    print(f"Max Solve Time: {max(solve_times):.3f}")
    print(f"Min Solve Time: {min(solve_times):.3f}")
    plt.show()
