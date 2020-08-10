import cdd
import numpy as np

class Polygon:
    def __init__(self, matrix):
        self.polyhedron = cdd.Polyhedron(matrix)

    @classmethod
    def from_a_b(cls, a_mat, b_mat):
        b_trans = np.reshape(b_mat, (b_mat.shape[0], 1))
        neg_a = np.negative(a_mat)
        b_neg_a = np.concatenate((b_trans, neg_a), axis=1)
        matrix = cdd.Matrix(b_neg_a, number_type='float')
        matrix.rep_type = cdd.RepType.INEQUALITY
        matrix.canonicalize()
        return cls(matrix)

    @classmethod
    def from_vertices(cls, vertices):
        ones = np.ones((vertices.shape[0], 1))
        ones_vertices = np.concatenate((ones, vertices), axis=1)
        matrix = cdd.Matrix(ones_vertices, number_type='float')
        matrix.rep_type = cdd.RepType.GENERATOR
        matrix.canonicalize()
        return cls(matrix)

    def get_b_neg_a(self):
        return np.array(self.polyhedron.get_inequalities())
        
    def get_a(self):
        b_neg_a = np.array(self.get_b_neg_a())
        return np.negative(b_neg_a[:,1:])

    def get_b(self):
        b_neg_a = np.array(self.get_b_neg_a())
        return b_neg_a[:,0]

    def get_vertices(self):
        gen = np.array(self.polyhedron.get_generators())
        only_vert = np.all(gen[:,0] == 1, axis = 0)
        if only_vert:
            return gen[:,1:]
        else:
            raise ValueError('Polygon contains rays.')

if __name__ == "__main__":
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    print("demo")
    poly = Polygon.from_vertices(np.array([[0, 0], [1, 0], [0, 1], [-0.2, 0.5]]))
    # poly = Polygon.from_a_b(np.array([[0, 0], [1, 0], [0, 1], [-0.2, 0.5]]), np.array([]))
    vert = poly.get_vertices()
    avg = np.average(vert, axis=0)
    def myfn(x):
        return np.arctan2(x[1] - avg[1], x[0] - avg[0])
    angles = np.array(map(myfn, vert))
    order = np.argsort(angles)
    vert_sorted = vert[order]
    vert = vert_sorted
    print("draw")
    fig, ax = plt.subplots()
    patch = mpatches.Polygon(vert, True)
    ax.add_patch(patch)
    ax.grid()
    ax.axis('equal')
    plt.show()

