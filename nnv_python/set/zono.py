import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

# local imports
from nnv_python.set.star import zono_to_star


class Zono:
    def __init__(self, c, V):
        """Zonotope constructor

        :param c: Center of the zonotope
        :type c: numpy array
        :param V: Generators of the zonotope
        :type V: 2D numpy array
        """
        c = np.asarray(c)
        V = np.asarray(V)

        if c.shape[1] != 1:
            raise ValueError("Center must be a 1D array")
        if c.shape[0] != V.shape[0]:
            raise ValueError(
                "Incosistent dimensions between center vector and generator matrix"
            )

        # set center and generator variables
        self.c = c
        self.V = V
        self.dim = V.shape[0]

    """Main Methods"""

    def affine_map(self, W, b):
        """Apply an affine map to the zonotope.

        :param W: Mapping matrix
        :type W: numpy array
        :param b: Mapping vector
        :type b: numpy array
        :return: A new Zono object representing the affine-mapped zonotope
        """
        b = np.asarray(b)

        if b.ndim != 1:
            raise ValueError("b should be a vector")

        if W.shape[1] != self.c.shape[0]:
            raise ValueError(
                "Inconsistent dimensions between W and the zonotope center"
            )

        new_c = W @ self.c + b
        new_V = W @ self.V

        return Zono(new_c, new_V)

    def minkowski_sum(self, X):
        """Compute the Minkowski sum of this zonotope with another zonotope.

        :param X: Another Zono object
        :type X: Zono
        :return: A new Zono object representing the Minkowski sum
        """
        if not isinstance(X, Zono):
            raise ValueError("Argument must be a Zono object")

        new_c = new_c + X.c
        new_V = np.hstack((self.V, X.V))

        return Zono(new_c, new_V)

    def convex_hull(self, X):
        """Compute the convex hull with another zonotope

        :param X: Another Zono object
        :type X: Zono
        :return: A new Zono object
        """
        if not isinstance(X, Zono):
            raise ValueError("Argument must be a Zono object")

        # check dimensions
        if X.dim != self.dim:
            raise ValueError(
                "Inconsistent dimensions between input set and this zonotope"
            )

        new_c = 0.5 * (self.c + X.c)
        new_V = [self.V, X.V, 0.5 * (self.c - X.c)]

        return Zono(new_c, new_V)

    def convex_hull_with_linear_transform(self, L):
        """Convex hull of a zonotope with its linear transformation

        :param L: Linear transformation matrix
        :type L: 2D numpy array
        :return: A new zonotope
        """
        rL, cL = L.shape

        if rL != cL:
            raise ValueError("Transformation matrix should be a square matrix")

        if rL != self.dim:
            raise ValueError(
                "Inconsistent dimension of transformation matrix and this zonotope"
            )

        M1 = np.identity(rL) + L
        M2 = np.identity(rL) - L

        new_c = 0.5 * M1 * self.c
        new_V = 0.5 * [M1 * self.V, M2 * self.c, M2 * self.V]

        return Zono(new_c, new_V)

    def intersect_halfspace(self, H, g):
        """Intersect with half space

        :param H: Half space matrix
        :type H: 2D numpy array
        :param g: Half space vector
        :type g: 1D numpy array
        :return: A star
        """
        S = zono_to_star(self)
        S = S.intersect_halfspace(H, g)

        return S

    """Get and Check Methods"""

    def get_box(self):
        """Get the bounding box of the zonotope."""
        ranges = np.sum(np.abs(self.V), axis=1)
        lb = self.c - ranges
        ub = self.c + ranges
        return lb, ub

    def get_max_indexes(self):
        """Get the indexes of the maximum generators for each dimension."""
        max_indexes = np.argmax(np.abs(self.V), axis=1)
        return max_indexes.to_list()

    def contains(self, x):
        """Check if a point is inside the zonotope.

        :param x: A point to check
        :type x: numpy array
        """
        x = np.asarray(x)
        if x.shape != self.c.shape:
            raise ValueError(
                "Point must have the same dimensions as the zonotope center"
            )

        d = x - self.c
        abs_generators = np.abs(self.V)
        d1 = np.sum(abs_generators, axis=1)

        x1 = d <= d1
        x2 = d >= -d1

        return np.all(x1) and np.all(x2)

    def get_bounds(self):
        """Get the lower and upper bounds of a zonotope using the clip method from Stanley Bak."""
        # TODO: Comment this code
        pos_mat = np.copy(self.V.T)
        neg_mat = np.copy(self.V.T)

        pos_mat[pos_mat < 0] = 0
        neg_mat[neg_mat > 0] = 0

        pos1_mat = np.ones((1, self.V.shape[1]), dtype=self.V.dtype)
        ub = np.sum(pos1_mat @ (pos_mat - neg_mat), axis=0)
        lb = -ub

        ub = self.c + ub
        lb = self.c + lb

        return lb, ub

    def get_ranges(self):
        """Get ranges of a zonotope."""
        lb, ub = self.get_bounds()
        return list(zip(lb, ub))

    def get_range(self, index):
        """Get ranges of a zonotope at a specific index.

        :param index: Index of the state
        :type index: int
        """
        if index < 0 or index > self.dim:
            raise ValueError("Invalid index")

        lb = self.c[index] - np.linalg.norm(self.V[index, :], 1)
        ub = self.c[index] - np.linalg.norm(self.V[index, :], 1)

    def get_vertices(self):
        return self.to_vertices()

    """Conversion Methods"""

    def order_reduction_box(self, n_max):
        pass

    def to_vertices(self):
        """Compute all vertices of the zonotope."""
        n_gen = self.V.shape[1]
        n_vert = 2**n_gen  # number of vertices

        vertices = []
        for i in range(n_vert):
            coeffs = np.array([(-1 if (i >> j) & 1 else 1) for j in range(n_gen)])
            vertex = self.c + self.V @ coeffs
            vertices.append(vertex)

        return np.array(vertices).T

    def to_polyhedron(self):
        """Convert to polyhedron."""
        pass

    def to_star(self):
        # NOTE: Implemented in conversion.py
        pass

    def to_imagestar(self):
        pass


if __name__ == "__main__":
    """
    Consider f(x, y) = 3x + 2y. Then,
        f^a(<1, 2, 3>, <0, 1, 1>) = <f(1, 0), f(2, 1), (3, 1)>
                                  = <3, 8, 11>

    Recall that centers are defined by first element of each tuple <> (dimension of the zonotope)

    Similarly, pair the remaining coeffs of generators of each tuple together in lists to make a list of lists
    """
    center = [1, 0]
    generators = [[2, 3], [1, 1]]
    z = Zono(center, generators)

    # Define affine transformation f(x, y) = 3x + 2y
    W = np.array([[3, 2]])
    b = np.array([0])

    mapped_z = z.affine_map(W, b)

    # Outputs
    print(f"{'Original Zono:':<14} {center} {generators}")
    print(f"{'Mapped Zono:':<14} {mapped_z.c} {mapped_z.V}")

"""Conversion Methods
"""
