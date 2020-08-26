"""Contains class for 2D polytopes."""

import cdd
import numpy as np

class Polytope:
    """
    Represents a 2D polytope.

    Example of creating a polytope that splits the 2D plane into two regions.
    ```
    # create a polytope from linear constraints
    poly = Polytope.from_a_b(np.array([
        [1, 1]
    ]), np.array([
        1
    ]))

    # create a polytope from vertices and rays
    poly = Polytope.from_vertices_rays(np.array([
        [0.5, 0.5]
    ]), np.array([
        [-1, 1],
        [1, -1]
    ]))
    ```

    Example of creating a square polytope centered at the origin.
    ```
    # create a polytope from linear constraints
    poly = Polytope.from_a_b(np.array([
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1]
    ]), np.array([
        1,
        1,
        1,
        1
    ]))

    # create a polytope from vertices and rays
    poly = Polytope.from_vertices_rays(np.array([
        [1, 1],
        [-1, 1],
        [1, -1],
        [-1, -1]
    ]))
    ```
    """

    def __init__(self, matrix): # noqa: D107
        self.polyhedron = cdd.Polyhedron(matrix)

    @classmethod
    def from_a_b(cls, a_mat, b_mat):
        """
        Create polytope from A and b matrices in the expression P={x|Ax<=b}.

        The polytope is canonicalized to remove all implicit linearities and all redundancies.

        Parameters
        ----------
        a_mat : numpy.array
            2D A matrix of the form [[a11, a12, ...], [a21, a22, ...], ...].
        b_mat : numpy.array
            1D b matrix of the form [b1, b2, ...].
        """
        b_trans = np.reshape(b_mat, (b_mat.shape[0], 1))
        neg_a = np.negative(a_mat)
        b_neg_a = np.concatenate((b_trans, neg_a), axis=1)
        matrix = cdd.Matrix(b_neg_a, number_type='float')
        matrix.rep_type = cdd.RepType.INEQUALITY
        matrix.canonicalize()
        return cls(matrix)

    @classmethod
    def from_vertices_rays(cls, vertices, rays=np.empty([0, 2])):
        """
        Create polytope from a set of vertices and rays.

        The polytope is canonicalized to remove all implicit linearities and all redundancies.
        The rays are vectors pointing towards vertices at infinity.

        Parameters
        ----------
        vertices : numpy.array
            Array of 2D vertices of the form [[x1, y1], [x2, y2], ...].
        rays : numpy.array
            Array of 2D rays of the form [[x1, y1], [x2, y2], ...].
        """
        ones = np.ones((vertices.shape[0], 1))
        ones_vertices = np.concatenate((ones, vertices), axis=1)
        zeros = np.zeros((rays.shape[0], 1))
        zeros_rays = np.concatenate((zeros, rays), axis=1)
        mat = np.concatenate((ones_vertices, zeros_rays), axis=0)
        matrix = cdd.Matrix(mat, number_type='float')
        matrix.rep_type = cdd.RepType.GENERATOR
        matrix.canonicalize()
        return cls(matrix)

    def get_a_b(self):
        """
        Return the polytope's A and b matrices.

        Returns
        ----------
        (a_mat, b_mat) : (numpy.array, numpy.array)
            A and b matrices.
        """
        b_neg_a = np.array(self.polyhedron.get_inequalities())
        a_mat = np.negative(b_neg_a[:, 1:])
        b_mat = b_neg_a[:, 0]
        return (a_mat, b_mat)

    def get_vertices_rays(self):
        """
        Return the polytope's vertices and rays.

        Returns
        ----------
        (vertices, rays) : (numpy.array, numpy.array)
            Vertices and rays.
        """
        gen = np.array(self.polyhedron.get_generators())
        rays = gen[gen[:, 0] == 0]
        verts = gen[gen[:, 0] == 1]
        def to_unit_vector(x):
            mag = np.sqrt(x[0]*x[0] + x[1]*x[1])
            return x/mag
        rays = np.array(map(to_unit_vector, rays[:, 1:]))
        verts = verts[:, 1:]
        return (verts, rays)

    def as_finite_polygon(self, ray_length=10):
        """
        Convert the polytope to a finite polygon.

        Parameters
        ----------
        ray_length : float
            Length to extend rays.

        Returns
        ----------
        vertices : numpy.array
            Array of vertices of the form [[x1, y1], [x2, y2], ...].
            The vertices are sorted in clockwise order.
        """
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
            def unit_vector(vec):
                mag = np.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                return vec/mag
            def bisector(a_vec, b_vec):
                a_hat = unit_vector(a_vec)
                b_hat = unit_vector(b_vec)
                c_vec = a_hat + b_hat
                return unit_vector(c_vec)
            def angle_vector(vec):
                return np.arctan2(vec[1], vec[0]) + np.pi
            # The following sorting and check attempt to place the rays on the
            # correct ends of the polygon. It may not correctly place the rays
            # for all polytopes.
            # ---
            # find a test point in the rays' region
            bisec = bisector(rays[0], rays[1])
            test_point = verts_avg + bisec * mag * 100
            # sort the vertices by angle to the test point
            # sort will be clockwise with index 0 on the clockwise side of the bisector
            def angle_at_test_point(vert):
                return -(
                    (angle_vector(vert - test_point) - angle_vector(bisec) + np.pi * 2)
                    % (np.pi * 2)
                )
            angles = np.array(map(angle_at_test_point, verts))
            order = np.argsort(angles)
            verts_sorted = verts[order]
            verts = verts_sorted
            # sort the rays by the sign of the angle to the bisector
            # sort will be clockwise from bisector
            order = np.array([
                (angle_vector(rays[1]) - angle_vector(bisec) + np.pi * 2) % (np.pi * 2),
                (angle_vector(rays[0]) - angle_vector(bisec) + np.pi * 2) % (np.pi * 2)
            ])
            rays = rays[np.argsort(order)]
            # add vertices to list for rays
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
