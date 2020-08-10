import cdd
import numpy as np

class Polytope:
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
        
    def get_a_b(self):
        b_neg_a = np.array(self.get_b_neg_a())
        a = np.negative(b_neg_a[:,1:])
        b = b_neg_a[:,0]
        return (a, b)

    def get_vertices_rays(self):
        gen = np.array(self.polyhedron.get_generators())
        rays = gen[gen[:,0] == 0]
        verts = gen[gen[:,0] == 1]
        def myfn(x):
            mag = np.sqrt(x[0]*x[0] + x[1]*x[1])
            return x/mag
        rays = np.array(map(myfn, rays[:,1:]))
        verts = verts[:,1:]
        return (verts, rays)

    def as_finite_polygon(self, ray_length=10):
        (verts, rays) = self.get_vertices_rays()
        # sort vertices by angle with the average of all vertices
        verts_avg = np.average(verts, axis=0)
        def angle_of_vertex(vert):
            return -np.arctan2(vert[1] - verts_avg[1], vert[0] - verts_avg[0])
        angles = np.array(map(angle_of_vertex, verts))
        order = np.argsort(angles)
        verts_sorted = verts[order]
        verts = verts_sorted
        # add vertices for rays
        mag = ray_length
        if rays.shape[0] == 2:
            start = [verts[0] + rays[0] * mag]
            center = rays[0] + rays[1]
            center = [(center / np.linalg.norm(center)) * mag + verts_avg]
            end = [verts[-1] + rays[1] * mag]
            verts = np.concatenate((start, verts), axis=0)
            verts = np.concatenate((verts, end), axis=0)
            verts = np.concatenate((verts, center), axis=0)
        elif rays.shape[0] == 1:
            start = [verts[0] + rays[0] * mag]
            center = [rays[0] * mag + verts_avg]
            end = [verts[-1] + rays[0] * mag]
            verts = np.concatenate((start, verts), axis=0)
            verts = np.concatenate((verts, end), axis=0)
            verts = np.concatenate((verts, center), axis=0)
        elif rays.shape[0] == 0:
            pass
        else:
            raise ValueError('Polytope contains more than two rays.')
        return verts

if __name__ == "__main__":
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    print("demo")
    poly = Polytope.from_vertices(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
    poly = Polytope.from_a_b(np.array([[1, 0], [0.4, -1], [-1, -0.1]]), np.array([1, 2, 2]))
    poly = Polytope.from_a_b(np.array([[-1, -0.1], [0.4, -1], [1, 0]]), np.array([2, 2, 1]))
    # poly = Polygon.from_a_b(np.array([[1, 0], [0.4, -1], [-1, -0.1]]), np.array([1, 2, 2]))
    verts = poly.as_finite_polygon(10)
    fig, ax = plt.subplots()
    patch = mpatches.Polygon(verts, True)
    ax.add_patch(patch)
    ax.grid()
    ax.axis('equal')
    plt.show()

