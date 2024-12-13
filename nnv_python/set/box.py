import numpy as np

# local imports
from nnv_python.set.zono import Zono
import nnv_python.set.conversion


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


if __name__ == "__main__":
    """
    Consider f(x, y) = 3x + 2y. Then,
        f([5, 10], [20, 30]) = [3*5 + 2*20, 3*10 + 2*30]
                             = [55, 90]
    """
    lb = [5, 20]  # lower bounds for x and y
    ub = [10, 30]  # upper bounds for x and y
    box = Box(lb, ub)

    # Define affine transformation f(x, y) = 3x + 2y
    W = np.array([[3, 2]])
    b = np.array([0])

    mapped_box = box.affine_map(W, b)

    # Outputs
    print(f"{'Original Box:':<14} {lb} {ub}")
    print(f"{'Mapped Box:':<14} {mapped_box.lb} {mapped_box.ub}")


"""Conversion Methods
"""
