
import numpy as np
import scipy.sparse as sparse
import time

import convert.misos_trajectory_problem as misos

def main():
    sd = misos.symmetric_determinant
    n = 20
    mat = np.random.random_integers(10, size=(n, n))
    # mat = sparse.rand(n, n, density=0.5, format="csc").toarray()
    for i in range(n):
        for j in range(i):
            mat[j, i] = mat[i, j]
    rng = [i for i in range(n)]
    start_time = time.time()
    det = sd(rng, rng, mat)

    print("np.linalg.det: {}\tmine: {}".format(np.linalg.det(mat), det))
    print(time.time()-start_time)

if __name__ == "__main__":
    main()
