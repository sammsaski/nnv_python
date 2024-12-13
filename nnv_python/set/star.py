import numpy as np
from scipy.optimize import linprog

# local imports
from nnv_python.set.box import Box

# from nnv_python.set.zono import Zono
import nnv_python.set.conversion


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
            ), "Inconsistent dimensions between V and C"
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

            if b.shape[1] != 1:
                raise ValueError("Mapping vector should have one column")

            new_V = W @ self.V
            new_V[:, 1] = new_V[:, 1] + b
        else:
            new_V = W @ self.V

        return Star(new_V, self.C, self.d, self.pred_lb, self.pred_ub)

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
            bounds=(self.pred_lb, self.pred_ub),
            method="highs",
        )
        res_max = linprog(
            -f,
            A_ub=-self.C,
            b_ub=-self.d,
            bounds=(self.pred_lb, self.pred_ub),
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
        pass

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


"""Conversion Methods
"""


def box_to_star(B):
    Z = box_to_zono(B)
    return zono_to_star(Z)


def zono_to_star(Z):
    n = Z.V.shape[1]
    lb = -np.ones((n, 1), dtype=Z.V.dtype)
    ub = np.ones((n, 1), dtype=Z.V.dtype)

    # create constraints
    C = np.vstack((np.identity(n, dtype=Z.V.dtype), -np.identity(n, dtype=Z.V.dtype)))
    d = np.ones((2 * n, 1), dtype=self.V.dtype)

    # construct the star
    S = Star(np.hstack((Z.C, Z.V)), C, d, lb, ub)

    return S


def imagestar_to_star(I):
    N = I.height * I.width * I.num_channel
    num_pred = I.V.shape[-1] - 1
    reshaped_V = I.V.reshape((N, num_pred + 1))

    if hasattr(I, "im_lb") and hasattr(self, "im_ub"):
        state_lb = I.im_lb.reshape((N, 1))
        state_ub = I.im_ub.reshape((N, 1))
        return Star(reshaped_V, I.C, I.d, I.pred_lb, I.pred_ub, state_lb, state_ub)

    return Star(reshaped_V, I.C, I.d, I.pred_lb, I.pred_ub)
