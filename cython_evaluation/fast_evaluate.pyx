cimport cython
from libc.math cimport exp, sin, pow

# @cython.boundscheck(False)
# @cython.weaparound(False)
cpdef eval(double[:] out, int x_num, double x_min, double x_max, double h_min, double h_max):
    """
    Evaluate
    """
    cdef double h, result, X, Y, Z

    for h_i in range(out.shape[0]):
        # find current h
        h = h_min + (h_max - h_min) * h_i / (out.shape[0] - 1)

        # where to save the results of evaluation
        result = 0.

        # Loop over x
        for x_i in range(x_num):
            X = x_min + (x_max - x_min) * x_i / (x_num - 1)

            # Loop over y
            for y_i in range(x_num):
                Y = x_min + (x_max - x_min) * y_i / (x_num - 1)

                # Loop over z
                for z_i in range(x_num):
                    Z = x_min + (x_max - x_min) * z_i / (x_num - 1)

                    result += exp(-pow(X - h, 2)) * (sin(Y + h) + pow(Z + h, 2))

        out[h_i] = result
