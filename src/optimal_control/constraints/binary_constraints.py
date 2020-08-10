"""Represent constraints involving binary variables."""


from optimal_control.constraints import constraint


class BinaryEqualityConstraint(constraint.EqualityConstraint):
    """Represent an equality constraint with binary variables."""
    
