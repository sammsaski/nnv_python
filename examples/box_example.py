import numpy as np

# local imports
from nnv_python.set.abstract import Box


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
