from nnv_python.set.zono import Zono


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
