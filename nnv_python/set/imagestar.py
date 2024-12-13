import numpy as np
from scipy.optimize import linprog

# local imports
from nnv_python.set.star import Star, imagestar_to_star


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
            if len(n) == 2 and len(l) == 2 and len(u) == 2:
                self.num_channel = 1
            else:
                self.num_channel = n[2]

            # set properties
            self.im = im
            self.lb = lb
            self.ub = ub
            self.height = n[0]
            self.width = n[1]

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
                    np.reshape(self.im_lb, (n[0] * n[1], 1)),
                    np.reshape(self.im_ub, (n[0] * n[1], 1)),
                )
                self.V = np.reshape(I.V, [n[0], n[1], 1, I.n_var + 1])

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
                self.height = n[0]
                self.width = n[1]
                self.num_channel = n[2]
            elif len(n) == 4:
                if n[3] != self.num_pred + 1:
                    raise ValueError(
                        "Consistency between the basis matrix and the number of predicate variables"
                    )
                else:
                    self.num_channel = n[2]
                    self.height = n[0]
                    self.width = n[1]
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
        new_V = np.dot(W, self.V)
        new_V[:, 0] += b

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
