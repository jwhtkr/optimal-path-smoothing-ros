# cspell: disable
"""Implement the MISOSTrajectoryProblem.m in python with pyomo."""
# import pyomo.environ as pyomo
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
        # check for mosek
        # check for yalmip

        if start.size[1] > self.traj_degree + 1:
            print('For a degree d polynomial, we can only constrain at most '
                  'the first d derivatives. Additional derivatives ignored.')
            start = start[:, :self.traj_degree]

        if goal.size[1] > self.traj_degree + 1:
            print('For a degree d polynomial, we can only constrain at most '
                  'the first d derivatives. Additional derivatives ignored.')
            goal = goal[:, :self.traj_degree]


        ########################################################################
        # The following (until the "closing" line of pound signs) is probably not
        # necessary in python. A Matlab `cell` is basically matlab's version of a
        # list, but also not quite. Point being, especially with Python's duck-typing
        # philosophy, these probably aren't needed.
        if not is_cell(safe_region_sets):
            safe_region_sets = {safe_region_sets}

        if not is_cell(safe_region_assignments):
            safe_region_assignments = {safe_region_assignments}
        ########################################################################

        dim = start.size[0]
        # nan should probably be replaced with None, depending on what exactly
        # size_check does. Not sure what `size_check` does though.
        size_check(goal, [dim, nan])

        C_BOUND = 100
        SIGMA_BOUND = 100

        # sdpvar (...) -> pyomo.Var(...) https://yalmip.github.io/command/sdpvar/
        t = sdpvar(1, 1)

        if self.basis == 'monomials':
            # https://yalmip.github.io/command/monolist/
            basis = monolist(t, self.traj_degree)
        elif self.basis == 'legendre':
            shifted_legendre = [1,
                                2*t - 1,
                                6*t^2 - 6*t + 1,
                                (20*t^3 - 30*t^2 + 12*t - 1)/5,
                                (70*t^4 - 140*t^3 + 90*t^2 - 20*t + 1)/10,
                                (252*t^5 - 630*t^4 + 560*t^3 - 210*t^2 + 30*t - 1)/50,
                                (924*t^6 - 2772*t^5 + 3150*t^4 - 1680*t^3 + 420*t^2 - 42*t + 1)/100,
                                (3432*t^7 - 12012*t^6 + 16632*t^5 - 11550*t^4 + 4200*t^3 - 756*t^2 + 56*t - 1)/1000]
            basis = shifted_legendre[0:self.traj_degree]
        else:
            print('Invalid basis name')

        assign(t, 0)
        basis_t0 = value(basis)
        assign(t, 1)
        basis_t1 = value(basis)

        c = []
        for j in range(0, self.num_traj_segments):
            c[j] = sdpvar(self.traj_degree+1, dim, 'full')  # sdpvar(...) -> pyomo.Var(...)
            x[j] = c[j]*basis
            # https://yalmip.github.io/command/jacobian/
            xd[j] = [jacobian(x[j], t)]
            # https://yalmip.github.io/command/coefficients/
            cd[j] = [[coefficients(xd[j][1][1], t), 0]]
            for d in range(1, dim):
                cd[j][0][:, d] = [coefficients(xd[j][0][d], t), 0]
            for k in range(1, self.traj_degree):
                xd[j][k] = jacobian(xd[j][k-1], t)
                cd[j][k] = [coefficients(xd[j][k][0], t), np.zeros((k, 1))]
                for d in range(1, dim):
                    cd[j][k][:, d] = [coefficients(xd[j][k][d], t), np.zeros((k, 1))]

        x0 = start[:, 1]
        xf = goal[:, 1]

        constraints = []

        for j in range(0, start.size[1]):
            if j == 0:
                constraints.append(c[0] * basis_t0 == start[:, j])  # constraints = [constraints, c[0] * basis_t0 == start[:,j]]
            else:
                constraints.append(c[0][j-1] * [1, zeros(self.traj_degree, 1)] == start[:, j]) # constraints = [constraints, c[0][j-1] * [1, zeros(self.traj_degree, 1)] == start[:,j]]

        for j in range(0, goal.size[1]):
            if j == 0:
                constraints.append(c[self.num_traj_segments] * basis_t1 == goal[:, j])  # constraints = [constraints, c[self.num_traj_segments] * basis_t1 == goal[:,j]]
            else:
                constraints.append(c[self.num_traj_segments][j-1] * [1, ones(self.traj_degree, 1)] == goal[:, j])  # constraints = [constraints, c[self.num_traj_segments][j-1] * [1, ones(self.traj_degree, 1)] == goal[:,j]]

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
