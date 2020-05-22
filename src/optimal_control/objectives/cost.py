"""Contains a base class for representing a cost function (to be minimized)."""


from optimal_control.objectives import objective


class Cost(objective.Objective):  # pylint: disable=abstract-method
    """
    Represent a cost function for minimization.

    See base class for attributes and parameters.
    """
    pass
