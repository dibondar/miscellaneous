import numpy as np
import time

######################################################################
#
#   Perform evaluation using Pyhton
#
######################################################################

x = np.linspace(-10, 10, 300)
X = x[:, np.newaxis, np.newaxis]
Y = x[np.newaxis, :, np.newaxis]
Z = x[np.newaxis, np.newaxis, :]

H = np.linspace(-5, 5, 10)

t0 = time.time()

result_python = np.array([
    np.sum(np.exp(-(2. * Z + X + Y ** 2- h)**2) * (np.sin(X + Y + 3.*Z + h) + (Y + Z + h) ** 2)) for h in H
])

print("running Python time: {} seconds".format(time.time() - t0))

######################################################################
#
#   Perform the same evaluation using Cython
#
#   Note: Do not forget to compile cython code running in the command line:
#       python3 setup.py build_ext --inplace
#
######################################################################

import fast_evaluate


result_cython = np.zeros_like(H)

t0 = time.time()

fast_evaluate.eval(result_cython, x.size, x.min(), x.max(), H.min(), H.max())

print("running Cython time: {} seconds".format(time.time() - t0))

assert np.allclose(result_python, result_cython), "results should be the same"

######################################################################
#
#   Perform the same evaluation using C shared library
#
#   Note: You must compile the C shared library
#       gcc -O3 -shared -o fastest_evaluate.so fastest_evaluate.c -lm -fopenmp
#
######################################################################

import os
import ctypes

# Load the shared library assuming that it is in the same directory
lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/fastest_evaluate.so")

# specify the parameters of the c-function
c_eval = lib.eval
c_eval.argtypes = (
    ctypes.POINTER(ctypes.c_double),    # double* out
    ctypes.c_int,                       # int size_out
    ctypes.c_int,                       # int x_num
    ctypes.c_double,                    # double x_min
    ctypes.c_double,                    # double x_max
    ctypes.c_double,                    # double h_min
    ctypes.c_double,                    # double h_max
)
c_eval.restype = ctypes.c_int

result_c = np.zeros_like(H)

t0 = time.time()

c_eval(
    result_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    result_c.size,
    x.size,
    x.min(),
    x.max(),
    H.min(),
    H.max()
)

print("running C-library time: {} seconds".format(time.time() - t0))

assert np.allclose(result_python, result_c), "results should be the same"