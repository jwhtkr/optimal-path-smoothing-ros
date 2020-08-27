# cspell: disable
"""Implement the MISOSTrajectoryProblem.m in python with pyomo."""
import pyomo.environ as pyo
import pyomo
import numpy as np

class MisosTrajectoryProblem:
    """Conversion of matlab class to python."""

    def __init__(self):  # noqa: D107
        self.traj_degree = 3
        self.num_traj_segments = 6
        self.bot_radius = 0
        self.basis = 'legendre'
        self.dt = 0.5
        self.debug = False

    def solve_trajectory(self, start, goal, safe_region_sets, safe_region_assignments=()):
        """Solve for a trajectory."""

        if start.shape[1] > self.traj_degree + 1:
            print('For a degree d polynomial, we can only constrain at most '
                  'the first d derivatives. Additional derivatives ignored.')
            start = start[:, :self.traj_degree]

        if goal.shape[1] > self.traj_degree + 1:
            print('For a degree d polynomial, we can only constrain at most '
                  'the first d derivatives. Additional derivatives ignored.')
            goal = goal[:, :self.traj_degree]

        dim = start.shape[0]
        if goal.shape[0] != dim:
            print('Goal and start are different sizes.')

        C_BOUND = 100
        SIGMA_BOUND = 100

        model = pyo.ConcreteModel()

        model.t = pyo.Var(within = pyo.Reals)
        t = model.t

        if self.basis == 'monomials':
            monomials = [1,
                         t,
                         t**2,
                         t**3,
                         t**4,
                         t**5,
                         t**6,
                         t**7]
            basis = monomials[0:self.traj_degree]
        elif self.basis == 'legendre':
            shifted_legendre = [1,
                                2*t - 1,
                                6*t**2 - 6*t + 1,
                                (20*t**3 - 30*t**2 + 12*t - 1)/5,
                                (70*t**4 - 140*t**3 + 90*t**2 - 20*t + 1)/10,
                                (252*t**5 - 630*t**4 + 560*t**3 - 210*t**2 + 30*t - 1)/50,
                                (924*t**6 - 2772*t**5 + 3150*t**4 - 1680*t**3 + 420*t**2 - 42*t + 1)/100,
                                (3432*t**7 - 12012*t**6 + 16632*t**5 - 11550*t**4 + 4200*t**3 - 756*t**2 + 56*t - 1)/1000]
            basis = shifted_legendre[0:self.traj_degree]
        else:
            print('Invalid basis name')

        t.set_value(0)
        basis_t0 = map(lambda x : pyo.value(x), basis)
        t.set_value(1)
        basis_t1 = map(lambda x : pyo.value(x), basis)

        print(basis_t0)
        print(basis_t1)

        a = range(0, self.num_traj_segments)
        b = range(0, self.traj_degree+1)
        c = range(0, dim)
        model.c = pyo.Var(a, b, c, within=pyo.Reals)

        # j - polynomial piece j
        # k - kth derivative
        # i - ith dimension?
        def gen_x(model, j, k, i):
            #? X{j} = C{j}'*basis
            return model.c[j, k, i] * basis[i]
        model.x = pyo.Expression(a, b, c, rule=gen_x)

        # i am not sure if this is correct for the jacobian
        def gen_xd(model, j, k, i):
            #? Xd{j} = {jacobian(X{j}, t)}
            return pyo.differentiate(model.x[j, k, i], wrt=t, mode=pyomo.core.expr.calculus.derivatives.Modes.sympy)
        model.xd = pyo.Expression(a, b, c, rule=gen_xd)

        # need Cd, created using coefficients(...) from xd
        


        x0 = start[:, 1]
        xf = goal[:, 1]

        # I don't think this does the same as the matlab
        def gen_init_c(model, j, k, i):
            if j == 0:
                return model.c[0, k, i] * basis_t0[i] == start[i, j]
            else:
                return model.cd[0, j-1, k, i] * 1 == start[i, j]
        model.init_c = pyo.Constraint(a, b, c, rule=gen_init_c)

        # I don't think this does the same as the matlab
        def gen_final_c(model, j, k, i):
            if j == 0:
                return model.c[self.num_traj_segments, k, i] * basis_t1[i] == goal[i, j]
            else:
                return model.cd[self.num_traj_segments, j-1, k, i] * 1 == goal[i, j]
        model.final_c = pyo.Constraint(a, b, c, rule=gen_init_c)


        region = []
        for j in range(0, len(safe_region_sets)):  # Convert to pythonic for loop (for safe_region_set in safe_region_sets: ...)
            if safe_region_sets[j].len > 1:
                if is_empty(cell_2_mat(safe_region_assignments)):  # python should probably be `if not safe_region_assignments: ...`
                    # binvar(...) -> pyomo.Var(..., within=pyomo.Boolean) https://yalmip.github.io/command/binvar/
                    region[j] = binvar(safe_region_sets[j].len, self.num_traj_segments, 'full')
                    constraints.append(sum(region[j], 1) == 1)  # constraints = [constraints, sum(region[j], 1) == 1]
                else:
                    region[j] = safe_region_assignments[j]
            else:
                # true(n1, n2, ...) is matlab-ese for make an array of all
                # `True` of shape (n1, n2, ...)
                region[j] = true(1, self.num_traj_segments)

        for j in range(0, self.num_traj_segments-1):
            constraints.append(c[j] * basis_t1 == c[j+1] * basis_t0)  # constraints = [constraints, c[j] * basis_t1 == c[j+1] * basis_t0]
            for d in range(0, self.traj_degree-1):
                constraints.append(cd[j][d] * [1, np.ones((self.traj_degree, 1))]
                                   == cd[j+1][d] * [1, np.zeros((self.traj_degree, 1))])
                # constraints = [constraints,
                #         cd[j][d] * [1, ones(self.traj_degree, 1)] == cd[j+1][d] * [1, zeros(self.traj_degree, 1)]]

        objective = 0

        sigma = []
        q = []
        m = monolist(t, (self.traj_degree-1)/2)
        for j in range(0, self.num_traj_segments):
            sigma[j] = []
            q[j] = []
            if is_empty(cell_2_mat(safe_region_assignments)):  # `if not safe_region_assignments` probably
                # This constraint, because it's double-sided, might need to be
                # split up into two for pyomo.
                constraints.append(-C_BOUND <= C[j] <= C_BOUND)  # constraints = [constraints, -C_BOUND <= C[j] <= C_BOUND]
            if self.traj_degree <= 3:
                for d in range(0, dim):
                    objective += 0.01 * (cd[j][self.traj_degree][:, d] * cd[j][self.traj_degree][:, d])
            elif self.traj_degree == 5:
                c = coefficients(x[j], t).reshape(-1, dim)
                objective += 0.01 * (sum((math.factorial(4) * c[end-1, :])^2 + 1/2 * 2 * (math.factorial(4) * c[end-1, :]) * (math.factorial(5) * c[end, :]) + 1/3 * (math.factorial(5) * c[end, :])^2))
            else:
                print('not implemented yet')

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
    problem.solve_trajectory(np.array([
        [0, 0],
    ]), np.array([
        [1, 1]
    ]), np.array([
        
    ]))
