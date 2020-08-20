class MisosTrajectoryProblem:
    def __init__(self):
        self.traj_degree = 3
        self.num_traj_segments = 6
        self.bot_radius = 0
        self.basis = 'legendre'
        self.dt = 0.5
        self.debug = False

    def solve_trajectory(self, start, goal, safe_region_sets, safe_region_assignments=[]):
        # check for mosek
        # check for yalmip
        
        if start.size[1] > self.traj_degree + 1:
             print('For a degree d polynomial, we can only constrain at most the first d derivatives. Additional derivatives ignored.')
             start = start[:,0:self.traj_degree]

        if goal.size[1] > self.traj_degree + 1:
             print('For a degree d polynomial, we can only constrain at most the first d derivatives. Additional derivatives ignored.')
             goal = goal[:,0:self.traj_degree]

        if !is_cell(safe_region_sets):
            safe_region_sets = {safe_region_sets}

        if !is_cell(safe_region_assignments):
            safe_region_assignments = {safe_region_assignments}

        dim = start.size[0]
        size_check(goal, [dim, nan])

        C_BOUND = 100
        SIGMA_BOUND = 100

        t = sdpvar(1,1)

        if self.basis = 'monomials':
            basis = monolist(t, self.traj_degree)
        else if self.basis = 'legendre':
            shifted_legendre = [1;
                              2*t - 1;
                              6*t^2 - 6*t + 1;
                              (20*t^3 - 30*t^2 + 12*t - 1)/5;
                              (70*t^4 - 140*t^3 + 90*t^2 - 20*t + 1)/10;
                              (252*t^5 - 630*t^4 + 560*t^3 - 210*t^2 + 30*t - 1)/50;
                              (924*t^6 - 2772*t^5 + 3150*t^4 - 1680*t^3 + 420*t^2 - 42*t + 1)/100;
                              (3432*t^7 - 12012*t^6 + 16632*t^5 - 11550*t^4 + 4200*t^3 - 756*t^2 + 56*t - 1)/1000;
                              ];
            basis = shifted_legendre(0:self.traj_degree)
        else:
            print('Invalid basis name')

        assign(t, 0)
        basis_t0 = value(basis)
        assign(t, 1)
        basis_t1 = value(basis)

        c = [];
        for j in range(0, self.num_traj_segments):
            c[j] = sdpvar(self.traj_degree+1, dim, 'full')
            x[j] = c[j]*basis
            xd[j] = [jacobian(x[j], t)]
            cd[j] = [[coefficients(xd[j][1][1], t), 0]]
            for d in range(1, dim):
                cd[j][0][:,d] = [coefficients(xd[j][0][d], t), 0]
            for k in range(1, self.traj_degree):
                xd[j][k] = jacobian(xd[j][k-1], t)
                cd[j][k] = [coefficients(xd[j][k][0], t), zeros(k, 1)]
                for d in range(1, dim):
                    cd[j][k][:,d] = [coefficients(xd[j][k][d], t), zeros(k, 1)]

        x0 = start[:,1]
        xf = goal[:,1]

        constraints = [];

        for j in range(0, start.size[1]):
            if j == 0:
                constraints = [constraints, c[0] * basis_t0 == start[:,j]]
            else:
                constraints = [constraints, c[0][j-1] * [1, zeros(self.traj_degree, 1)] == start(:,j)]

        for j in range(0, goal.size[1]):
            if j == 0:
                constraints = [constraints, c[self.num_traj_segments] * basis_t1 == goal[:,j]]
            else:
                constraints = [constraints, c[self.num_traj_segments][j-1] * [1, ones(self.traj_degree, 1)] == goal[:,j]]

        region = [];
        for j in range(0, safe_region_sets.len):
            if safe_region_sets[j].len > 1:
                if is_empty(cell_2_mat(safe_region_assignments)):
                    region[j] = binvar(safe_region_sets[j].len, self.num_traj_segments, 'full')
                    constraints = [constraints, sum(region[j], 1) == 1]
                else:
                    region[j] = safe_region_assignments[j]
            else:
                region[j] = true(1, self.num_traj_segments)

        for j in range(0, self.num_traj_segments-1):
            constraints = [constraints, c[j] * basis_t1 == c[j+1] * basis_t0];
            for d in range(0, self.traj_degree-1):
                constraints = [constraints,
                        cd[j][d] * [1, ones(self.traj_degree, 1)] == cd[j+1][d] * [1, zeros(self.traj_degree, 1)]]

        objective = 0

        sigma = []
        q = []
        m = monolist(t, (self.traj_degree-1)/2)
        for j in range(0, self.num_traj_segments):
            sigma[j] = []
            q[j] = []
            if is_empty(cell_2_mat(safe_region_assignments)):
                constraints = [constraints, -C_BOUND <= C[j] <= C_BOUND]
            if self.traj_degree <= 3:
                for d in range(0, dim):
                    objective = objective + 0.01 * (cd[j][self.traj_degree][:,d] * cd[j][self.traj_degree][:,d])
            else if self.traj_degree == 5:
                c = reshape(coefficients(x[j], t), [], dim)
                objective = objective + 0.01 * (sum((factorial(4) * c[end-1,:])^2 + 1/2 * 2 * (factorial(4) * c[end-1,:]) * (factorial(5) * c[end,:]) + 1/3 * (factorial(5) * c[end,:])^2))
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
                        ai = a[k,:]
                        bi = b[k]
                        n = norm(ai)
                        ai = ai / n
                        bi = bi / n
                        [coeff, ...] = coefficients(bi - self.bot_radius - (ai*(c[j]) * basis), t, monolist(t, obj.traj_degree))
                        if k > sigma[j][rs].len:
                            sigma[j][rs][k] = [sdpvar(1, self.traj_degree), sdpvar(1, self.traj_degree)]
                            if self.traj_degree == 1:
                                constraints = [constraints,
                                        sigma[j][rs][k][0] >= 0,
                                        sigma[j][rs][k][1] >= 0]
                            else if self.traj_degree == 3:
                                constraints = [constraints,
                                        rcone(sigma[j][rs][k][0][1], 2*sigma[j][rs][k][0][0], sigma[j][rs][k][0][2]),
                                        rcone(sigma[j][rs][k][1][1], 2*sigma[j][rs][k][1][0], sigma[j][rs][k][1][2])]
                            else if self.traj_degree > 3:
                                q[j][rs][k] = [sdpvar((self.traj_degree-1)/2 + 1), sdpvar((self.traj_degree-1)/2 + 1)]
                                constraints = [constraints,
                                        sigma[j][rs][k][0] == coefficients(m*q[j][rs][k][0]*m, t),
                                        sigma[j][rs][k][1] == coefficients(m*q[j][rs][k][1]*m, t),
                                        q[j][rs][k][0] >= 0,
                                        q[j][rs][k][1] >= 0]
                            else:
                                print('not implemented')
                            
                            if isa(region[rs][0, 0], 'sdpvar'):
                                constraints = [constraints,
                                        -SIGMA_BOUND <= sigma[j][rs][k][0] <= SIGMA_BOUND,
                                        -SIGMA_BOUND <= sigma[j][rs][k][1] <= SIGMA_BOUND]

                        if isa(region[rs][r, j], 'sdpvar'):
                            constraints = [constraints,
                                    implies(region[rs][r,j], coeff == [0, sigma[j][rs][k]] + [sigma[j][rs][k], 0] - [0, sigma[j][rs][k][1]])]
                        else if region[rs][r,j]:
                            constraints = [constraints,
                                    coeff == [0, sigma[j][rs][k][0]] + [sigma[j][rs][k][1], 0] - [0, sigma[j][rs][k][1]]]

        t0 = tic;
        if self.traj_degree > 3 && is_empty(cell_2_mat(safe_region_assignments)):
            diagnostics = optimize(constraints, objective, sdpsettings('solver', 'bnb', 'bnb.maxiter', 5000, 'verbose', 1, 'debug', True))
        else:
            # mosek optimize
        toc(t0)

        breaks = 0:1:self.num_traj_segments
        coeffs = zeros([dim, breaks.len-1, self.traj_degree])

        if diagnostics.problem == 0 || diagnostics.problem == 9:
            for k in range(0, breaks.len-1):
                c = value(c[k])
                for i in range(0, dim):
                    [ct, ...] = coefficients(c[:,i] * basis, t, monolist(t, self.traj_degree))
                    if ct.len < self.traj_degree + 1:
                        ct = [ct; 1e-6]
                    coeffs[i, k, :] = fliplr(reshape(ct, 1, []))

        objective = double(objective)

        ytaj = PPTrajectory(mkpp(breaks, coeffs, dim))


                            


