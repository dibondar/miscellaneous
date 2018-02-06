import numpy as np
import time

######################################################################
#
#   Perform evaluation using Pyhton
#
######################################################################

x = np.linspace(-10, 10, 100)
X = x[:, np.newaxis, np.newaxis]
Y = x[np.newaxis, :, np.newaxis]
Z = x[np.newaxis, np.newaxis, :]

H = np.linspace(-5, 5, 10)

t0 = time.time()

result_python = np.array([
    np.sum(np.exp(-(2. * Z + X + Y ** 2- h)**2) * (np.sin(X + Y + 3.*Z + h) + (Y + Z + h) ** 2)) for h in H
])

print("funning time: {} seconds".format(time.time() - t0))

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

print("funning time: {} seconds".format(time.time() - t0))

assert np.allclose(result_python, result_cython), "results should be the same"
