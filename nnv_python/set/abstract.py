import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


class Box:
    def __init__(self, lb, ub):
        """
        Constructor for the Box class.

        :param lb: Lower-bound vector (numpy array)
        :param ub: Upper-bound vector (numpy array)
        """
        lb = np.asarray(lb)
        ub = np.asarray(ub)

        if lb.shape[1] != 1 or ub.shape[1] != 1:
            raise ValueError("lb and ub should be a vector")

        if lb.shape[0] != ub.shape[0]:
            raise ValueError("Inconsistent dimensions between lb and ub")

        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        self.center = 0.5 * (lb + ub)
        vec = 0.5 * (ub - lb)

        self.generators = []  # Initialize generators

        try:
            # Speeding up implementation using diagonal matrix
            gens = np.diag(vec.flatten())  # Generate matrix

            if gens.size > 1:
                # Delete columns with no information
                gens = gens[:, ~(gens == 0).all(axis=0)]

            self.generators = gens
        except MemoryError:
            # This works well for large input sets with few perturbed pixels
            self.generators = np.zeros((self.dim, 0), dtype=ub.dtype)
            gen_locs = np.where(vec != 0)[0]
            for i in gen_locs:
                gen = np.zeros(self.dim, dtype=ub.dtype)
                gen[i] = vec[i]
                self.generators.append(gen)

        self.generators = np.array(self.generators).T  # Convert list to matrix

    # Methods
    def affine_map(self, W, b):
        """Perform an affine mapping of the box.

        :param W: Mapping matrix
        :type W: numpy array
        :param b: Mapping vector
        :type b: numpy array
        """
        b = np.asarray(b)

        if b.ndim != 1:
            raise ValueError("b should be a vector")

        if W.shape[0] != b.shape[0]:
            raise ValueError("Inconsistency between mapping matrix and mapping vector")

        new_center = W @ self.center + b
        new_generators = W @ self.generators

        n = len(new_center)
        new_lb = np.zeros(n, dtype=W.dtype)
        new_ub = np.zeros(n, dtype=W.dtype)

        for i in range(n):
            v = new_generators[i, :]
            new_lb[i] = new_center[i] - np.linalg.norm(v, ord=1)
            new_ub[i] = new_center[i] + np.linalg.norm(v, ord=1)

        return Box(new_lb, new_ub)

    def get_range(self):
        """Get the lower and upper bounds of the box.

        :return: A tuple containing lb and ub.
        :rtype: tuple
        """
        return self.lb, self.ub

    def get_vertices(self):
        """Get all vertices of the box.

        :return: A 2D numpy array where each column is a vertex of the box
        :rtype: numpy array
        """
        n = len(self.lb)
        N = 2**n  # number of vertices in the worst case
        V = []

        for i in range(N - 1):
            b = f"{i:0{n}b}"  # Binary representation of i, padded to n bits
            v = np.zeros(n, dtype=self.lb.dtype)

            for j in range(n):
                if b[j] == "1":
                    v[j] = self.ub[j]
                else:
                    v[j] = self.lb[j]

            V.append(v)

        # Delete duplicate vertices
        V = np.unique(V, axis=0).T

        return V

    @staticmethod
    def box_hull(boxes):
        """Merge multiple boxes into one box that bounds them all.

        :param boxes: List of Box objects
        :type boxes: list
        :return: A new Box object bounding all input boxes
        """
        if not boxes:
            raise ValueError("Input list of boxes cannot be empty.")

        lb = np.min([box.lb for box in boxes], axis=0)
        ub = np.max([box.ub for box in boxes], axis=0)

        return Box(lb, ub)

    def to_polyhedron():
        pass

    def to_star(self):
        # NOTE: Implemented in conversion.py
        pass

    def to_zono(self):
        # NOTE: Implemented in conversion.py
        pass


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


class Star:
    """
    A star set is defined by:
        x = c + a[1]*v[1] + a[2]*v[2] + ... + a[n]*v[n]
          = V * b, V = [c v[1] v[2] ... v[n]]
        b = [1 a[1] a[2] ... a[n]]^T
    subject to the constraints:
        C * a <= d
    """

    def __init__(self, *args):
        """Constructor for a star.

        Arguments:
            V
            C
            d
            pred_lb
            pred_ub
            state_lb
            state_ub

        :param args: _description_
        :type args: _type_
        """
        if len(args) == 7:
            V = args[0]
            C = args[1]
            d = args[2]
            pred_lb = args[3]
            pred_ub = args[4]
            state_lb = args[5]
            state_ub = args[6]

            self.V = np.asarray(V)
            self.C = np.asarray(C) if C is not None else np.empty((0, 0))
            self.d = np.asarray(d) if d is not None else np.empty((0,))
            self.pred_lb = np.asarray(pred_lb) if pred_lb is not None else None
            self.pred_ub = np.asarray(pred_ub) if pred_ub is not None else None
            self.state_lb = np.asarray(state_lb) if state_lb is not None else None
            self.state_ub = np.asarray(state_ub) if state_ub is not None else None
            self.dim = self.V.shape[0]
            self.n_var = self.C.shape[1]

            self._validate_dimensions()

        elif len(args) == 2:
            lb = args[0]
            ub = args[1]

            B = Box(lb, ub)
            S = box_to_star(B)
            self.V = S.V
            self.C = np.zeros((1, S.n_var), dtype=lb.dtype)
            self.d = np.zeros((1, 1), dtype=lb.dtype)
            self.dim = S.dim
            self.n_var = S.n_var
            self.state_lb = lb
            self.state_ub = ub
            self.pred_lb = -np.ones((S.n_var, 1), dtype=lb.dtype)
            self.pred_ub = np.ones((S.n_var, 1), dtype=lb.dtype)

    def _validate_dimensions(self):
        """Validate dimensions of V, C, and d."""
        if self.C.size > 0:
            assert (
                self.V.shape[1] == self.C.shape[1] + 1
            ), f"Inconsistent dimensions between V {self.V.shape} and C {self.C.shape}"
        if self.d.size > 0:
            assert (
                self.C.shape[0] == self.d.size
            ), "Inconsistent dimensions between C and d"

        if self.pred_lb is not None and self.pred_ub is not None:
            assert (
                self.pred_lb.shape == self.pred_ub.shape
            ), "Predicate bounds must have the same shape"
            assert (
                self.pred_lb.size == self.C.shape[1]
            ), "Predicate bounds size mismatch with constraints"

        if self.state_lb is not None and self.state_ub is not None:
            assert (
                self.state_lb.shape == self.state_ub.shape
            ), "State bounds must have the same shape"
            assert (
                self.state_lb.size == self.V.shape[0]
            ), "State bounds size mismatch with state variables"

    """Main Methods"""

    def affine_map(self, W, b):
        """Affine mapping of a star set.

        :param W: Mapping matrix
        :type W: 2D numpy array
        :param b: Mapping vector
        :type b: 1D numpy array
        """
        if W.shape[1] != self.dim:
            raise ValueError(
                "Inconsistency between the affine mapping matrix and dimension of the star set"
            )

        if not b.size == 0:  # if not empty
            if b.shape[0] != W.shape[0]:
                raise ValueError(
                    "Inconsistency between the mapping vec and mapping matrix"
                )

            if len(b.shape) > 1:
                if b.shape[1] != 1:
                    raise ValueError("Mapping vector should have one column")

            new_V = W @ self.V
            new_V[:, 0] += b
        else:
            new_V = W @ self.V

        # Reduce the number of generators in new_V to ensure consistency with the dimensionality of the new space
        expected_generators = W.shape[0]
        if new_V.shape[1] != expected_generators + 1:
            new_V = new_V[:, : expected_generators + 1]

        # Transform the constraints
        if self.C is not None:
            W_pinv = np.linalg.pinv(W)
            new_C = self.C @ W_pinv
            new_d = self.d + self.C @ (W_pinv @ b)

            # Transform predicate bounds
            if self.pred_lb is not None and self.pred_ub is not None:
                # Transform bounds using new_V (basis matrix in new space)
                new_pred_lb = np.min(new_V[:, 1:], axis=1)
                new_pred_ub = np.max(new_V[:, 1:], axis=1)
            else:
                new_pred_lb, new_pred_ub = None, None

        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub, None, None)

    def relu(self):
        new_lb = self.pred_lb.copy()
        new_ub = self.pred_ub.copy()

        for i in range(len(self.pred_lb)):
            if self.pred_ub[i] <= 0:  # inactive neurons
                new_lb[i] = 0
                new_ub[i] = 0
            elif self.pred_lb[i] < 0 < self.pred_ub[i]:  # cross-zero case
                # solve LP to refine bounds
                res = linprog(
                    self.V[:, i],
                    A_ub=self.C,
                    b_ub=self.d,
                    bounds=(self.pred_lb[i], self.pred_ub[i]),
                    method="highs",
                )
                if res.success:
                    new_lb[i] = res.fun
                else:
                    raise ValueError(
                        f"Optimization failed for index {i}: {res.message}"
                    )

                res = linprog(
                    -self.V[:, i],
                    A_ub=self.C,
                    b_ub=self.d,
                    bounds=(self.pred_lb[i], self.pred_ub[i]),
                    method="highs",
                )
                if res.success:
                    new_ub[i] = res.fun
                else:
                    raise ValueError(
                        f"Optimization failed for index {i}: {res.message}"
                    )
                new_lb[i] = max(0, new_lb[i])  # clamp lower bound to 0

        return Star(
            self.V, self.C, self.d, new_lb, new_ub, self.state_lb, self.state_ub
        )

    def minkowski_sum(self, X):
        pass

    def hadamard_product(self, X):
        pass

    def scalar_map(self, alp_max):
        """Perform a scalar map of the star, S' = alp * S, 0 <= alp <= alp_max

        # TODO: Write better docstring

        :param alp_max: Maximum value of a
        :type alp_max: int
        """
        assert alp_max >= 0, "alp_max must be non-negative"

        # New basic matrix with additional column for the scalar mapping
        new_c = np.zeros((self.V.shape[0], 1), dtype=self.V.dtype)
        new_V = np.hstack((self.V, new_c))

        # New constraint matrix and vector
        new_C = np.vstack(
            (
                np.hstack(
                    (np.zeros((self.C.shape[0], 1)), self.C)
                ),  # Original constraints
                np.array([-1, 1]).reshape(2, 1),  # Scalar mapping constraints
            )
        )

        new_d = np.concatentate((alp_max * self.d, np.array([0, alp_max])))

        # Scale predicate bounds
        new_pred_lb = None
        new_pred_ub = None
        if self.pred_lb is not None and self.pred_ub is not None:
            new_pred_lb = self.pred_lb * alp_max
            new_pred_ub = self.pred_ub * alp_max

        # Scale state bounds
        new_state_lb = None
        new_state_ub = None
        if self.state_lb is not None and self.state_ub is not None:
            new_state_lb = self.state_lb * alp_max
            new_state_ub = self.state_ub * alp_max

        return Star(
            new_V, new_C, new_d, new_pred_lb, new_pred_ub, new_state_lb, new_state_ub
        )

    def intersect_halfspace(self, H, g):
        """Intersection of a star with a half space."""
        rH, cH = H.shape
        rg, cg = g.shape

        if cg != 1:
            raise ValueError("Halfspace vector should have one column")

        if rH != rg:
            raise ValueError(
                "Inconsistent dimensions between halfspace matrix and halfspace vector"
            )

        if cH != self.dim:
            raise ValueError("Inconsistent dimension between halfspace and star set")

        m = self.V.shape[1]
        C1 = H @ self.V[:, 2:m]
        d1 = g - H @ self.V[:, 1]

        new_C = np.vstack(self.C, C1)
        new_d = np.vstack(self.d, d1)

        S = Star(
            self.V,
            new_C,
            new_d,
            self.pred_lb,
            self.pred_ub,
            self.state_lb,
            self.state_ub,
        )

        if S._is_empty_set():
            S = []

        return S

    """Get and Check Methods"""

    def get_box(self):
        """Get the box bounding the star"""
        # star set is just a vector (one point)
        if self.C.size == 0 or self.d.size == 0:  # check if empty
            lb = self.V[:, 0]
            ub = self.V[:, 0]

        # star set is a set
        else:
            lb = np.zeros(self.V.shape[0])
            ub = np.zeros(self.V.shape[0])

            for i in range(self.V.shape[0]):
                f = self.V[i, 1 : self.n_var + 1]
                if np.all(f == 0):
                    lb[i] = self.V[i, 1]
                    ub[i] = self.V[i, 1]
                else:
                    res = linprog(
                        f,
                        A_ub=self.C,
                        b_ub=self.d,
                        bounds=(self.pred_lb, self.pred_ub),
                        method="highs",
                    )
                    if res.success:
                        lb[i] = res.fun + self.V[i, 0]
                    else:
                        return None  # Failed to find bounds

                    res = linprog(
                        -f,
                        A_ub=self.C,
                        b_ub=self.d,
                        bounds=(self.pred_lb, self.pred_ub),
                        method="highs",
                    )
                    if res.success:
                        ub[i] = -res.fun + self.V[i, 0]
                    else:
                        return None  # Failed to find bounds

        return Box(lb, ub)

    def get_box_fast(self):
        """Quickly estimate a box bounding the star (over-approximation)."""
        pred_lb, pred_ub = self.get_predicate_bounds()
        center = self.V[:, 0]
        ranges = np.sum(np.abs(self.V[:, 1:]), axis=1)
        lb = center - ranges
        ub = center + ranges
        return Box(lb, ub)

    def get_max_indexes(self):
        """Return possible max indexes for the state variables."""
        pass

    def get_predicate_bounds(self):
        """Get the bounds of the predicate variables."""
        if self.pred_lb is not None and self.pred_ub is not None:
            return self.pred_lb, self.pred_ub

        # Approximate bounds using Polyhedron
        res_lb = linprog(
            np.zeros(self.C.shape[1]), A_ub=self.C, b_ub=self.d, method="highs"
        )
        res_ub = linprog(
            np.zeros(self.C.shape[1]), A_ub=-self.C, b_ub=-self.d, method="highs"
        )

        if res_lb.success and res_ub.success:
            return res_lb.x, res_ub.x
        else:
            raise RuntimeError("Cannot compute predicate bounds")

    def get_range(self, index):
        """Find the range of the state variable at a specific index.

        :param index: _description_
        :type index: _type_
        """
        if index < 0 or index >= self.V.shape[0]:
            raise ValueError("Index out of range")

        f = self.V[index, 1:]
        if np.all(f == 0):
            return self.V[index, 0], self.V[index, 0]

        res_min = linprog(
            f,
            A_ub=self.C,
            b_ub=self.d,
            bounds=(self.pred_lb[index], self.pred_ub[index]),
            method="highs",
        )
        res_max = linprog(
            -f,
            A_ub=-self.C,
            b_ub=-self.d,
            bounds=(self.pred_lb[index], self.pred_ub[index]),
            method="highs",
        )

        if res_min.success and res_max.success:
            return res_min.fun + self.V[index, 0], -res_max.fun + self.V[index, 0]
        else:
            raise RuntimeError("Cannot compute range for the given index")

    def get_min(self, index):
        """Get the minimum value of the state variable at a specific index.

        :param index: _description_
        :type index: _type_
        :param lp_solver: _description_, defaults to "linprog"
        :type lp_solver: str, optional
        """
        f = self.V[index, 1:]
        if np.all(f == 0):
            return self.V[index, 0]

        res = linprog(
            f,
            a_ub=self.C,
            b_ub=self.d,
            bounds=(self.pred_lb, self.pred_ub),
            method="highs",
        )
        if res.success:
            return res.fun + self.V[index, 0]
        else:
            raise RuntimeError(f"Cannot find minimum, linprog failed: {res.message}")

    def get_mins(self, indices):
        """Get the minimum values of the state variables at specific indices.

        :param indices: _description_
        :type indices: _type_
        :param par_option: _description_, defaults to "single"
        :type par_option: str, optional
        :param dis_option: _description_, defaults to []
        :type dis_option: list, optional
        :param lp_solver: _description_, defaults to "linprog"
        :type lp_solver: str, optional
        """
        return [self.getMin(i) for i in indices]

    def get_max(self, index, lp_solver="linprog"):
        pass

    def get_maxs(self, map, par_option="single", dis_option=[], lp_solver="linprog"):
        pass

    def get_ranges(self):
        """Get ranges"""
        n = self.dim
        lb = np.zeros((n, 1))
        ub = np.zeros((n, 1))

        for i in range(n):
            temp_lb, temp_ub = self.get_range(i)
            lb[i] = temp_lb
            ub[i] = temp_ub

        return lb, ub

    def estimate_range(self, index):
        pass

    def estimate_bound(self, index):
        pass

    def estimate_bounds(self):
        pass

    def get_max_point_candidates(self):
        pass

    def get_oriented_box(self):
        pass

    def get_zono(self):
        pass

    def contains(self, s):
        """Check if the star set contains a point

        :param s: _description_
        :type s: _type_
        """
        if s.shape != (self.dim,):
            raise ValueError("Dimension mismatch")

        A = self.C
        b = self.d
        Ae = self.V[:, 1:]
        be = s - self.V[:, 0]

        res = linprog(
            np.zeros(Ae.shape[1]),
            a_ub=A,
            b_ub=b,
            A_eq=Ae,
            b_eq=be,
            bounds=(self.pred_lb, self.pred_ub),
            method="highs",
        )
        return res.success

    def is_p1_larger_than_p2(self, p1_id, p2_id):
        """Check if the value at index p1_id can be larger than at p2_id."""
        if not (0 <= p1_id < self.dim and 0 <= p2_id < self.dim):
            raise ValueError("Index out of range")

        d1 = self.V[p1_id, 0] - self.V[p2_id, 0]
        C1 = self.V[p2_id, 1:] - self.V[p1_id, 1:]
        new_C = np.vstack([self.C, C1])
        new_d = np.hstack([self.d, d1])

        S = Star(
            self.V,
            new_C,
            new_d,
            self.pred_lb,
            self.pred_ub,
            self.state_lb,
            self.state_ub,
        )
        return not S._is_empty_set()

    """Conversion Methods"""

    def convex_hull(self, X):
        """Compute the convex hull of this star and another star

        :param X: _description_
        :type X: _type_
        """
        pass

    def convex_hull_with_linear_transform(self, L):
        pass

    def to_polyhedron(self):
        pass

    def to_imagestar(self, height, width, numchannel):
        pass

    def to_volumestar(self, height, width, depth, numchannel):
        pass

    def reset_row(self, map):
        pass

    def scale_row(self, map, gamma):
        pass

    def concatenate(self, X):
        pass

    def concatenate_with_vector(self, v):
        pass

    """Helper Methods"""

    def _is_empty_set(self):
        """Check if the star set is empty.

        :return: True if the set is empty, False otherwise
        """
        f = np.zeros(self.C.shape[1], dtype=self.V.dtype)  # Objective function

        # Use linprog to solve the linear program
        result = linprog(
            c=f,
            A_ub=self.C,
            b_ub=self.d,
            bounds=(self.pred_lb, self.pred_ub),
            method="highs",
        )

        if result.success:
            return False  # Feasible solution exists, set is not empty
        elif result.status in [2, 3]:
            return True  # Problem is infeasible or unbounded, set is empty
        else:
            raise RuntimeError(f"Unexpected linprog result: {result.message}")


class ImageStar:
    def __init__(self, *args):
        """Initialize an ImageStar."""
        # self.V = np.asarray(V)
        # self.C = np.asarray(C)
        # self.d = np.asarray(d)
        # self.pred_lb = np.asarray(pred_lb) if pred_lb is not None else None
        # self.pred_ub = np.asarray(pred_ub) if pred_ub is not None else None

        if (
            len(args) == 3
        ):  # input center image and lower and upper bound matrices (box-representation)
            im = args[0]
            lb = args[1]
            ub = args[2]

            n = im.shape
            l = lb.shape
            u = ub.shape

            if n[0] != l[0] or n[0] != u[0] or n[1] != l[1] or n[1] != u[1]:
                raise ValueError(
                    "Inconsistency between center image and attack bound matrices"
                )

            if len(n) != len(l) or len(n) != len(u):
                raise ValueError(
                    "Inconsistency between center image and attack bound matrices"
                )

            # handle grayscale/rgb images
            if n[0] == 1 and l[0] == 1 and u[0] == 1:
                self.num_channel = 1
            else:
                self.num_channel = n[2]

            # set properties
            self.im = im
            self.lb = lb
            self.ub = ub
            self.height = n[1]
            self.width = n[2]

            self.im_lb = im - lb
            self.im_ub = im + ub

            # convert box imagestar to an array of 2D stars
            n = self.im_lb.shape
            if len(n) == 3:
                I = Star(
                    np.reshape(self.im_lb, (n[0] * n[1] * n[2], 1)),
                    np.reshape(self.im_ub, (n[0] * n[1] * n[2], 1)),
                )
                self.V = np.reshape(I.V, [n[0], n[1], n[2], I.n_var + 1])
            else:
                I = Star(
                    np.reshape(self.im_lb, (n[1] * n[2], 1)),
                    np.reshape(self.im_ub, (n[1] * n[2], 1)),
                )
                self.V = np.reshape(I.V, [n[1], n[2], 1, I.n_var + 1])

            self.C = I.C
            self.d = I.d
            self.pred_lb = I.pred_lb
            self.pred_ub = I.pred_ub
            self.num_pred = I.n_var

        elif len(args) == 5:
            self.V = args[0]
            self.C = args[1]
            self.d = args[2]
            self.pred_lb = args[3]
            self.pred_ub = args[4]

            if self.C.shape[0] != self.d.shape[0]:
                raise ValueError(
                    "Inconsistent dimension between constraint matrix and constraint vector"
                )

            if self.d.shape[1] != 1:
                raise ValueError(
                    "Invalid constraint vector, vector should have one column"
                )

            self.num_pred = self.C.shape[1]

            if (
                self.C.shape[1] != self.pred_lb.shape[0]
                or self.C.shape[1] != self.pred_ub.shape[0]
            ):
                raise ValueError(
                    "Number of predicates is different from the size of the lower bound or upper bound predicate vector"
                )

            if self.pred_lb.shape[1] != 1 or self.pred_ub.shape[1] != 1:
                raise ValueError(
                    "Invalid lower/upper bound predicate vector, vector should have one column"
                )

            n = self.V.shape

            if len(n) == 3:
                self.height = n[1]
                self.width = n[2]
                self.num_channel = n[0]
            elif len(n) == 4:
                if n[3] != self.num_pred + 1:
                    raise ValueError(
                        "Consistency between the basis matrix and the number of predicate variables"
                    )
                else:
                    self.num_channel = n[0]
                    self.height = n[1]
                    self.width = n[2]
            else:
                raise ValueError("Invalid basis matrix")

    """Main Methods"""

    def affine_map(self, scale, offset):
        """Affine mapping of an ImageStar.

        :param scale: _description_
        :type scale: _type_
        :param offset: _description_
        :type offset: _type_
        """
        if (
            scale is not None
            and np.isscalar(scale)
            and scale.shape[2] != self.num_channel
        ):
            raise ValueError(
                "Inconsistent number of channels between scale array and the ImageStar"
            )

        print("scale shape:", scale.shape)
        print("V shape:", self.V.shape)

        # Apply scaling
        if scale is not None:
            if scale.ndim == 0 or scale.size == 1:
                new_V = scale * self.V
            elif scale.ndim == 3 and scale.shape[2] == self.num_channel:
                new_V = np.multiply(self.V, scale)
            else:
                new_V = np.einsum(
                    "ij,kljm->klim", scale, self.V
                )  # matrix multiplication
        else:
            new_V = self.V.copy()

        # Apply offset to the first basis slice
        if offset is not None:
            new_V[..., 0] += offset

        return ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)

    def minkowski_sum(self, I):
        pass

    def concatenation(self, I):
        """Concatenate this ImageStar with another ImageStar

        :param I: _description_
        :type I: _type_
        """
        if not isinstance(I, ImageStar):
            raise TypeError("Input must be an ImageStar")

        if (
            self.height != I.height
            or self.width != I.width
            or self.num_channel != I.num_channel
        ):
            raise ValueError(
                "ImageStars must have the same dimensions for concatenation."
            )

        new_V = np.concatenate((self.V, I.V), axis=1)
        new_C = np.block(
            [
                [self.C, np.zeros((self.C.shape[0], self.I.shape[1]))],
                [np.zeros((I.C.shape[0], self.C.shape[1])), I.C],
            ]
        )
        new_d = np.concatenate((self.d, I.d))

        return ImageStar(new_V, new_C, new_d, self.pred_lb, self.pred_ub)

    def reshape_imagestar(self, target_dim):
        """Reshape the ImageStar

        :param target_dim: _description_
        :type target_dim: _type_
        """
        pass

    def upsample(self, scale_dim):
        pass

    def hadamard_product(self, I):
        pass

    def project_2D(self, point1, point2):
        pass

    """Get and Check Methods"""

    def is_empty_set(self):
        """Check if the ImageStar represents an empty set"""
        f = np.zeros(self.C.shape[1])
        res = linprog(
            f,
            A_ub=self.C,
            b_ub=self.d,
            bounds=(self.pred_lb, self.pred_ub),
            method="highs",
        )
        return not res.success

    def contains(self, image):
        """Check if the ImageStar contains a given given.

        :param image: _description_
        :type image: _type_
        """
        n = image.shape
        if len(n) == 2:
            if n[0] != self.height or n[1] != self.width or n[2] != 1:
                raise ValueError(
                    "Inconsistent dimensions between input image and the ImageStar"
                )
            y = np.reshape(image, (n[0] * n[1], 1))
        elif len(n) == 3:
            if n[0] != self.height or n[1] != self.width or n[2] != self.num_channel:
                raise ValueError(
                    "Inconsistent dimensions between input image and the ImageStar"
                )
            y = np.reshape(image, (n[0] * n[1] * n[2], 1))
        else:
            raise ValueError("Invalid input image")

        S = imagestar_to_star(self)
        return S.contains(y)

    def get_range(self, h, w, c):
        """Get the range of a specific index in the ImageStar"""
        if self.C.size == 0 or self.d.size == 0:
            raise ValueError("The ImageStar is empty")

        if h < 1 or h > self.height:
            raise ValueError("Invalid vertical index")

        if w < 1 or w > self.width:
            raise ValueError("Invalid horizontal index")

        if c < 1 or c > self.num_channel:
            raise ValueError("Invalid channel index")

        # get min

        f = self.V(h, w, c, list(range(1, self.num_pred + 1)))
        center = self.V[h, w, c, 0]
        min_res = linprog(
            f,
            A_ub=self.C,
            b_ub=self.d,
            bounds=(self.pred_lb, self.pred_ub),
            method="highs",
        )

        # get max
        max_res = linprog(
            -f,
            A_ub=self.C,
            b_ub=self.d,
            bounds=(self.pred_lb, self.pred_ub),
            method="highs",
        )

        if min_res.success and max_res.success:
            return center + min_res.fun, center - max_res.fun
        else:
            raise RuntimeError("Unable to compute range")

    def get_ranges(self, lp_solver="linprog"):
        """Get the ranges for all indices in the ImageStar.

        :param lp_solver: _description_, defaults to 'linprog'
        :type lp_solver: str, optional
        :return: _description_
        :rtype: _type_
        """
        ranges = [self.get_range(i) for i in range(self.V.shape[0])]
        return ranges

    def estimate_range(self, h, w, c):
        """Estimate the range of a specific index using predicate bounds.

        :param h: _description_
        :type h: _type_
        :param w: _description_
        :type w: _type_
        :param c: _description_
        :type c: _type_
        """
        f = self.V[index, 1:]
        center = self.V[h, w, c, 0]
        min_val = center + np.dot(f, self.pred_lb)
        max_val = center + np.dot(f, self.pred_ub)
        return min_val, max_val

    def estimate_range_fast(self, vert_ind, horiz_ind, chan_ind):
        pass

    def estimate_ranges(self, dis_opt=[]):
        """Estimate the ranges for all indices using predicate bounds.

        :param dis_opt: _description_, defaults to []
        :type dis_opt: list, optional
        """
        ranges = [self.estimate_range(i) for i in range(self.V.shape[0])]
        return ranges

    def get_num_attacked_pixels(self):
        pass

    """Conversion"""

    def to_star(self):
        # NOTE: Implemented in conversion.py
        pass

    """Helper Functions"""

    def is_max():
        pass

    def reshape(imagestar_in, new_shape):
        pass

    def add_constraint(imagestar_in, p1, p2):
        pass


"""Conversion Methods
"""


def box_to_zono(B):
    return Zono(B.center, B.generators)


def box_to_star(B):
    Z = box_to_zono(B)
    return zono_to_star(Z)


def zono_to_star(Z):
    n = Z.V.shape[1]
    lb = -np.ones((n, 1), dtype=Z.V.dtype)
    ub = np.ones((n, 1), dtype=Z.V.dtype)

    # create constraints
    C = np.vstack((np.identity(n, dtype=Z.V.dtype), -np.identity(n, dtype=Z.V.dtype)))
    d = np.ones((2 * n, 1), dtype=Z.V.dtype)

    # construct the star
    S = Star(np.hstack((Z.c, Z.V)), C, d, lb, ub, None, None)

    return S


def imagestar_to_star(I):
    N = I.height * I.width * I.num_channel
    num_pred = I.V.shape[-1] - 1
    reshaped_V = I.V.reshape((N, num_pred + 1))

    if hasattr(I, "im_lb") and hasattr(I, "im_ub"):
        state_lb = I.im_lb.reshape((N, 1))
        state_ub = I.im_ub.reshape((N, 1))
        return Star(reshaped_V, I.C, I.d, I.pred_lb, I.pred_ub, state_lb, state_ub)

    return Star(reshaped_V, I.C, I.d, I.pred_lb, I.pred_ub)
