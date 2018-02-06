import numpy as np

######################################################################
#
#   Perform evaluation using Pyhton
#
######################################################################

x = np.linspace(-10, 10, 10)
X = x[:, np.newaxis, np.newaxis]
Y = x[np.newaxis, :, np.newaxis]
Z = x[np.newaxis, np.newaxis, :]

H = np.linspace(-5, 5, 10)

result_python = np.array([
    np.sum(np.exp(-(X - h)**2) * (np.sin(Y + h) + (Z + h) ** 2)) for h in H
])

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
fast_evaluate.eval(result_cython, x.size, x.min(), x.max(), H.min(), H.max())

assert np.allclose(result_python, result_cython), "results should be the same"
