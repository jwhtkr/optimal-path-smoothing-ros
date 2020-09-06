# force floating point division as default
from __future__ import division

# cspell: disable
"""Implement the MISOSTrajectoryProblem.m in python with pyomo."""
import mosek
import pyomo.environ as pyo
import pyomo.core.expr.current as expr
import pyomo
import numpy as np
import matplotlib.pyplot as plt

class MisosTrajectoryProblem:
    """Conversion of matlab class to python."""

    def __init__(self):  # noqa: D107
        self.traj_degree = 3 # degree of polynomial used
        self.num_traj_segments = 6 # number of segments in the path
        self.bot_radius = 0
        self.basis = 'legendre' # set of basis polynomials to use
        self.dt = 0.5
        self.debug = False

    def solve_trajectory(self, start, goal, safe_regions, safe_region_assignments=None):
        """Solve for a trajectory."""

        # start - array of dimensions, of derivative constraints. (starts with position)
        # goal - array of dimensions, of derivative constraints. (starts with position)
        # safe_regions - array of A b matrx objects

        # 2D position: start = [[px], [py]]
        # 2D position, velocity: start = [[px, vx], [py, vy]]
        # 3D position, velocity: start = [[px, vx], [py, vy], [pz, vz]]

        # constant bounds
        C_BOUND = 100
        SIGMA_BOUND = 100

        # number of coefficients in the polynomial
        coefficient_num = self.traj_degree + 1
        print('')
        print('coefficient_num')
        print(coefficient_num)

        # === Check Inputs ===

        # Remove extra derivatives that can not be
        # constrained by the degree of polynomial selected.
        if start.shape[1] > coefficient_num:
            print('For a degree d polynomial, we can only constrain at most '
                  'the first d derivatives. Additional derivatives ignored.')
            start = start[:, 0:coefficient_num]

        # Remove extra derivatives that can not be
        # constrained by the degree of polynomial selected.
        if goal.shape[1] > coefficient_num:
            print('For a degree d polynomial, we can only constrain at most '
                  'the first d derivatives. Additional derivatives ignored.')
            goal = goal[:, 0:coefficient_num]

        # Check the start and goal have the same dimension.
        dim = start.shape[0]
        if goal.shape[0] != dim:
            print('Goal and start have different dimensions.')

        print('')
        print('start')
        print(start)
        print('')
        print('goal')
        print(goal)

        # === Construct Coefficients ===
        
        # Start with a basis coefficients matrix b_mat.
        # basis = b_mat * [1 t t^2 t^3 ...]
        if self.basis == 'monomials':
            monomials = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
            ])
            b_mat = monomials[0:coefficient_num, 0:coefficient_num]
        elif self.basis == 'legendre':
            shifted_legendre = np.array([
                [      1,       0,         0,         0,           0,          0,           0,         0],
                [     -1,       2,         0,         0,           0,          0,           0,         0],
                [      1,      -6,         6,         0,           0,          0,           0,         0],
                [   -1/5,    12/5,     -30/5,      20/5,           0,          0,           0,         0],
                [   1/10,  -20/10,     90/10,   -140/10,       70/10,          0,           0,         0],
                [  -1/50,   30/50,   -210/50,    560/50,     -630/50,     252/50,           0,         0],
                [  1/100, -42/100,   420/100, -1680/100,    3150/100,  -2772/100,     924/100,         0],
                [-1/1000, 56/1000, -765/1000, 4200/1000, -11550/1000, 16632/1000, -12012/1000, 3432/1000]
            ])
            b_mat = shifted_legendre[0:coefficient_num, 0:coefficient_num]
        else:
            print('Invalid basis name')

        print('')
        print('b_mat')
        print(b_mat)

        # === Setup Derivative Coefficients ===
        # This part is very different from how the matlab source does it.
        # Here a single derivative list for the basis coefficient matrix is created.
        # These derivative coefficients can be combined with pyomo variables to generate the correct coefficient expressions.

        # j - 0..path segments
        # C - coefficients for each path segment: array path segments long, of array dimensions long, of vector of coefficients
        # coefficient vectors for each segement for each dimension
        # X - C * basis, is coefficients using basis polynomials
        # Xd - coefficients of derivatives of each polynomial

        # differentiation matrix used to generate derivatives of polynomials in the monomial basis: coef_mat * [1 t t^2 t^3 ...]
        # another basis can also be used by grouping the basis matrix with the coefficients in the basis: (coef_mat * b_mat) * [1 t t^2 t^3 ...]
        # coef_mat * derivative_mat * [1 t t^2 t^3 ...] is the derivative of the polynomial coef_mat * [1 t t^2 t^3 ...]
        poly_derivative = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 5, 0, 0, 0],
            [0, 0, 0, 0, 0, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 7, 0],
        ])
        derivative_mat = poly_derivative[0:coefficient_num, 0:coefficient_num]
        
        # generates the k first derivatives including the 0th derivative of the polynomial with given coefficients
        def derivatives(poly_coef, num):
            poly_coef_deriv = poly_coef
            for _ in range(num+1):
                yield poly_coef_deriv
                poly_coef_deriv = np.matmul(poly_coef_deriv, derivative_mat)

        print('')
        print('derivative_mat')
        print(derivative_mat)

        # # array of derivatives for the basis, C[k] is the kth derivative
        Cd = [x for x in derivatives(b_mat, self.traj_degree)]

        print('')
        print('Cd')
        print(Cd)

        # find derivatives of basis at t=0 and t=1
        basis_derivatives_t0 = [np.matmul(x, [1, 0, 0, 0, 0, 0, 0, 0][0:coefficient_num]) for x in derivatives(b_mat, self.traj_degree)]
        basis_derivatives_t1 = [np.matmul(x, [1, 1, 1, 1, 1, 1, 1, 1][0:coefficient_num]) for x in derivatives(b_mat, self.traj_degree)]

        print('')
        print('basis_derivatives_t0')
        print(basis_derivatives_t0)
        print('')
        print('basis_derivatives_t1')
        print(basis_derivatives_t1)

        # === Create Pyomo Model ===

        # contains all elements that will be considered by the model
        model = pyo.ConcreteModel()

        # === Create Variable Coefficient Matrix ===

        # Matrix of polynomial coefficients, C[a, b, c] is the a-th segment, b-th dimension, c-th coefficient
        a = range(self.num_traj_segments) # range of segments
        b = range(dim) # range of dimensions
        c = range(coefficient_num) # range of coefficients
        model.C = pyo.Var(a, b, c, within=pyo.Reals)

        # === Basic Constraints ===
        # because pyomo does not support matrix operations, the following implements the matrix multiplications directly

        # constrain the initial position and derivatives using given start
        # performs the matrix multiplication for each derivative: model.C * basis_0t == start[:, 0]
        k = range(start.shape[1]) # range of constrained derivatives
        print('')
        print('init_derivative_constraints')
        def init_derivative_constraint(model, k, b):
            ex = expr.SumExpression([model.C[0, b, i] * basis_derivatives_t0[k][i] for i in c]) == start[b, k]
            print(ex.to_string())
            return ex
        model.init_derivative_constraints = pyo.Constraint(k, b, rule=init_derivative_constraint)

        # constrain the final position and derivatives using given start
        # performs the matrix multiplication for each derivative: model.C * basis_1t == goal[:, 0]
        k = range(goal.shape[1]) # range of constrained derivatives
        print('')
        print('final_derivative_constraints')
        def final_derivative_constraint(model, k, b):
            ex = expr.SumExpression([model.C[self.num_traj_segments-1, b, i] * basis_derivatives_t1[k][i] for i in c]) == goal[b, k]
            print(ex.to_string())
            return ex
        model.final_derivative_constraints = pyo.Constraint(k, b, rule=final_derivative_constraint)

        # enforce the continuity of the derivatives between segments
        # performs the matrix multiplication for each derivative: model.C * basis_1t == model.C * basis_0t
        k = range(self.traj_degree) # range of all derivatives except the final one
        print('')
        print('continuity_constraints')
        def continuity_constraint(model, a, k, b):
            ex = (expr.SumExpression([model.C[a, b, i] * basis_derivatives_t1[k][i] for i in c])
                == expr.SumExpression([model.C[a+1, b, i] * basis_derivatives_t0[k][i] for i in c]))
            print(ex.to_string())
            return ex
        model.continuity_constraints = pyo.Constraint(range(self.num_traj_segments - 1), k, b, rule=continuity_constraint)

        # === Region Assignments ===
        # simplified from the matlab to always use pyomo variables for the complete H matrix as in the paper

        # if safe_region_assignments is None:
        #     # creates variables for assigning each polynomial for each dimension to a single region
        #     r = range(len(safe_regions)) # range of all safe regions
        #     model.region = pyo.Var(a, b, r, within=pyo.Binary)
        #
        #     # constrain each polynomial to be assigned to a single region
        #     def single_region_constraint(model, a, b):
        #         return expr.SumExpression([model.region[a, b, x] for x in r]) == 1
        #     model.single_region_constraints = pyo.Constraint(a, b, rule=single_region_constraint)

        # === Bound Coefficients ===

        if safe_region_assignments is None:
            def coefficient_bound_constraint(model, a, b, c):
                return pyo.inequality(-C_BOUND, model.C[a, b, c], C_BOUND)
            model.coefficient_bound_constraints = pyo.Constraint(a, b, c, rule=coefficient_bound_constraint)

        # === Objective Function ===

        if self.traj_degree <= 3:
            # minimize 3rd derivative
            obj_fn = 0.01*expr.SumExpression([expr.SumExpression([model.C[i, j, l] * Cd[3][l, k] for l in c])**2 for i in a for j in b for k in c])
            print('')
            print('obj_fn')
            print(obj_fn)
            model.obj = pyo.Objective(expr = obj_fn)
        elif self.traj_degree == 5:
            # minimize 4th derivative
            obj_fn = 0.01*expr.SumExpression([expr.SumExpression([model.C[i, j, l] * Cd[4][l, k] for l in c])**2 for i in a for j in b for k in c])
            print('')
            print('obj_fn')
            print(obj_fn)
            model.obj = pyo.Objective(expr = obj_fn)
        else:
            print("Not implemented yet")

        # === Run Program ===
        
        # test objective function, sum all coefficients in all polynomials
        # model.obj = pyo.Objective(expr = expr.SumExpression([model.C[i, j, k] for i in a for j in b for k in c]))

        # test solve using mosek as solver
        opt = pyomo.opt.SolverFactory('mosek')
        opt.solve(model)

        print('')
        print('values')
        print(model.C.get_values())
        print('')
        print('obj_fn')
        print(pyo.value(obj_fn))

        # plot found solution

        for l in a:
            coeffs = [model.C[l, 0, x].value for x in c]
            x = np.linspace(0, 1, 100)
            y = np.array([np.sum(np.array([coeffs[i]*(j**i) for i in range(len(coeffs))])) for j in x])
            x = np.linspace(l, l+1, 100)
            plt.plot(x, y)
        plt.show()

        print("done")

        return

        objective = 0

        sigma = []
        q = []
        m = monolist(t, (self.traj_degree-1)/2)
        for j in range(0, self.num_traj_segments):
            sigma[j] = []
            q[j] = []

            for rs in range(0, region.len):
                sigma[j][rs] = []
                q[j][rs] = []
                nr = ragion[rs].size[0]
                for r in range(0, nr):
                    a = safe_region_sets[rs][r].a
                    b = safe_region_sets[rs][r].b
                    for k in range(0, a.size[0]):
                        ai = a[k, :]
                        bi = b[k]
                        n = norm(ai)
                        ai = ai / n
                        bi = bi / n
                        coeff, _ = coefficients(bi - self.bot_radius - (ai*(c[j]) * basis), t, monolist(t, obj.traj_degree))
                        if k > sigma[j][rs].len:
                            # sdpvar(...) -> pyomo.Var(...)
                            sigma[j][rs][k] = [sdpvar(1, self.traj_degree), sdpvar(1, self.traj_degree)]
                            if self.traj_degree == 1:
                                constraints.extend([sigma[j][rs][k][0] >= 0,
                                                    sigma[j][rs][k][1] >= 0])
                                # constraints = [constraints,
                                #                sigma[j][rs][k][0] >= 0,
                                #                sigma[j][rs][k][1] >= 0]
                            elif self.traj_degree == 3:
                                constraints = [constraints,
                                               # https://yalmip.github.io/command/rcone/
                                               # for info on `rcone()`.
                                               # I might have to be the one to
                                               # implement this helper function
                                               # if pyomo doesn't have something
                                               # similar.
                                               rcone(sigma[j][rs][k][0][1], 2*sigma[j][rs][k][0][0], sigma[j][rs][k][0][2]),
                                               rcone(sigma[j][rs][k][1][1], 2*sigma[j][rs][k][1][0], sigma[j][rs][k][1][2])]
                            elif self.traj_degree > 3:
                                q[j][rs][k] = [sdpvar((self.traj_degree-1)/2 + 1), sdpvar((self.traj_degree-1)/2 + 1)]
                                constraints.extend([sigma[j][rs][k][0] == coefficients(m*q[j][rs][k][0]*m, t),
                                                    sigma[j][rs][k][1] == coefficients(m*q[j][rs][k][1]*m, t),
                                                    q[j][rs][k][0] >= 0,
                                                    q[j][rs][k][1] >= 0])
                                # constraints = [constraints,
                                #                sigma[j][rs][k][0] == coefficients(m*q[j][rs][k][0]*m, t),
                                #                sigma[j][rs][k][1] == coefficients(m*q[j][rs][k][1]*m, t),
                                #                q[j][rs][k][0] >= 0,
                                #                q[j][rs][k][1] >= 0]
                            else:
                                print('not implemented')

                            if isa(region[rs][0, 0], 'sdpvar'):
                                # These double-sided constraints might also need
                                # to be split up.
                                constraints.extend([-SIGMA_BOUND <= sigma[j][rs][k][0] <= SIGMA_BOUND,
                                                    -SIGMA_BOUND <= sigma[j][rs][k][1] <= SIGMA_BOUND])
                                # constraints = [constraints,
                                #         -SIGMA_BOUND <= sigma[j][rs][k][0] <= SIGMA_BOUND,
                                #         -SIGMA_BOUND <= sigma[j][rs][k][1] <= SIGMA_BOUND]

                        if isa(region[rs][r, j], 'sdpvar'):
                            # https://yalmip.github.io/command/implies/
                            constraints.append(implies(region[rs][r, j],
                                                       coeff == [0, sigma[j][rs][k]] + [sigma[j][rs][k], 0] - [0, sigma[j][rs][k][1]]))
                            # constraints = [constraints,
                            #         implies(region[rs][r,j], coeff == [0, sigma[j][rs][k]] + [sigma[j][rs][k], 0] - [0, sigma[j][rs][k][1]])]
                        elif region[rs][r, j]:
                            coefficients.append(coeff == [0, sigma[j][rs][k][0]] + [sigma[j][rs][k][1], 0] - [0, sigma[j][rs][k][1]])
                            # constraints = [constraints,
                            #         coeff == [0, sigma[j][rs][k][0]] + [sigma[j][rs][k][1], 0] - [0, sigma[j][rs][k][1]]]

        t0 = time.time()  # tic
        # `and not safe_region_assignments:...` probably
        if self.traj_degree > 3 and is_empty(cell_2_mat(safe_region_assignments)):
            diagnostics = optimize(constraints, objective, sdpsettings('solver', 'bnb', 'bnb.maxiter', 5000, 'verbose', 1, 'debug', True))
        else:
            # mosek optimize
            pass
        print("Elapsed time: {}".format(time.time() - t0))  # toc(t0)

        breaks = range(self.num_traj_segments)  # 0:1:self.num_traj_segments
        coeffs = np.zeros(([dim, len(breaks)-1, self.traj_degree]))

        if diagnostics.problem == 0 or diagnostics.problem == 9:
            for k in range(len(breaks)-1):  # for i, break in enumerate(breaks[:-1])?
                c = value(c[k])
                for i in range(dim):
                    ct, _ = coefficients(c[:, i] * basis, t, monolist(t, self.traj_degree))
                    if len(ct) < self.traj_degree + 1:
                        ct = np.concatenate([ct, 1e-6])
                    # I have no idea what `fliplr(...)` is
                    coeffs[i, k, :] = fliplr(ct.reshape(1, -1))

        objective = float(objective)

        ytraj = PPTrajectory(mkpp(breaks, coeffs, dim))

        # btw, you probably already know this, but there is still a little more
        # after the debug section of the Matlab file that I think should be here.

if __name__ == "__main__":
    problem = MisosTrajectoryProblem();
    problem.num_traj_segments = 5
    problem.traj_degree = 3
    problem.basis = 'monomials'
    problem.solve_trajectory(np.array([
        # [-2.4, -2.4],
        [-2.4, 0.],
    ]), np.array([
        # [-3.2, 0.5],
        [-3.2, 0.],
    ]), np.array([
        
    ]))
