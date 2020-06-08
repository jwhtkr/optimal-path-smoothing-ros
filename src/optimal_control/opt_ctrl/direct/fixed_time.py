"""Contains classes to represent fixed terminal time direct optimal constrol."""


import optimal_control.opt_ctrl.direct.direct_optimal_control as dir_opt_ctrl


class FixedTime(dir_opt_ctrl.DirectOptimalControl):
    """
    Represent a direct optimal control problem with a fixed terminal time.

    The fixed terminal time indicates that the final time (and for convenience,
    at least for now, the temporal step size as well) is not a variable subject
    to optimization. Thus the number of steps, `n_step`, along with either the
    final time or a fixed step size, determines the discretization and time
    horizon of the problem. For now, the initial time is considered to be 0.

    Attributes
    ----------
    n_step : int
        The number of time steps in the optimized time interval.
    t_final : double
        The fixed end time of the optimization.
    step_size : double
        The fixed step size of discretization.
    See base class for further attribute descriptions.

    Parameters
    ----------
    See base class for first parameter descriptions.
    n_step : int
        The number of time steps in the optimized time interval.
    t_final : double, optional
        The fixed end time of the optimization. May be left unspecified if
        `time_step` is given.
    time_step : double, optional
        The fixed step size of discretization. May be left unspecified if
        `t_final` is given.
    **kwargs
        Keyword arguments to be passed to the `setup` function of the solver.

    """
    def __init__(self, dynamics, constraints, cost, solver, n_step,
                 t_final=0., time_step=0., **kwargs):
        self.n_step = n_step
        if t_final <= 0.:
            t_final = time_step*n_step
        if time_step <= 0.:
            time_step = t_final/n_step
        if not t_final or not time_step:
            raise TypeError("At least one of t_final or step size must be "
                            "specified. Any non-negative value counts as being "
                            "unspecified.")
        self.t_final = t_final
        self.time_step = time_step

        super(FixedTime, self).__init__(dynamics, constraints, cost, solver,
                                        **kwargs)

    def update(self, **kwargs):
        # TODO: Add/move functionality here that is general enough.
        super(FixedTime, self).update(**kwargs)
