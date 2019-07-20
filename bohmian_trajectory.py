########################################################################################################################
#
#   Bohmian trajectories from data
#
########################################################################################################################

import numpy as np
from scipy.interpolate import RectBivariateSpline
from split_op_schrodinger1D import SplitOpSchrodinger1D, njit
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.io import loadmat

########################################################################################################################
#
#   get the wavefunction by simulations
#
########################################################################################################################

sys = SplitOpSchrodinger1D(

    x_grid_dim=512,
    x_amplitude=20,

    v=njit(lambda x, t: 0.1 * x ** 2),
    k=njit(lambda p, t: 0.5 * p ** 2),
    dt=0.02
).set_wavefunction(
    lambda x: np.exp(-0.2 * (x + 8) ** 2) + np.exp(-0.2 * (x + 2) ** 2)
)

wavefunctions = [sys.wavefunction.copy()]
wavefunctions.extend(sys.propagate(15).copy() for _ in range(20))
wavefunctions = np.array(wavefunctions)[:, ::5]

x = sys.x[::5]
t = 15 * sys.dt * np.arange(0, wavefunctions.shape[0])

########################################################################################################################
#
#   get the wavefunction from file
#
########################################################################################################################
#
# data = loadmat('matlab_ws_7_WF.mat')
# x = data['t'].reshape(-1)
#
# wavefunctions = np.hstack(
#     [data['psi' + str(_)] for _ in range(1, 8)]
# ).T
# t = 0.07 * np.arange(wavefunctions.shape[0])
#
# # make sure normalize the wavefunctions
# wavefunctions /= (x[1] - x[0]) * np.linalg.norm(wavefunctions, axis=1)[:, np.newaxis] ** 2

########################################################################################################################
#
#   Interpolate
#
########################################################################################################################

# 2D spline for the real and imaginary parts of the wavefunction: psi = psi(t, x)
psi_real = RectBivariateSpline(t, x, wavefunctions.real)
psi_imag = RectBivariateSpline(t, x, wavefunctions.imag)

# interpolated values
test_t = np.linspace(t.min(), t.max(), 20 * t.size)
test_x = np.linspace(x.min(), x.max(), 3 * x.size)
w = psi_real(test_t, test_x) + 1j * psi_imag(test_t, test_x)

########################################################################################################################
#
#   Propagate Bohmian particle
#
########################################################################################################################

def bohmian_velocity(t, x):
    v = psi_real(t, x, 0, 1) + 1j * psi_imag(t, x, 0, 1)
    v /= psi_real(t, x) + 1j * psi_imag(t, x)
    return v.imag.reshape(-1)

########################################################################################################################
#
#   Propagate the Bohmian trajectory
#
########################################################################################################################

r = ode(bohmian_velocity)
r.set_initial_value(
    # np.hstack([np.linspace(9.5, 11, 6), np.linspace(12.5, 13.5, 4)]),
    np.linspace(-10, 0, 15),
    test_t[0]
)

t_trajectory = []
x_trajectory = []

for t_instant in test_t[1:]:

    if not r.successful():
        raise ValueError("Problem with propagator")

    r.integrate(t_instant)

    t_trajectory.append(r.t)
    x_trajectory.append(r.y)

########################################################################################################################
#
#   Plot
#
########################################################################################################################

extent = [x.min(), x.max(), t.min(), t.max()]

imag_params = dict(
    extent=extent,
    origin='lower',
    aspect=(extent[1] - extent[0]) / (extent[-1] - extent[-2]),

    # cmap='seismic',
)

# plt.subplot(121)

# plt.title('Original')

plt.imshow(np.abs(wavefunctions) ** 2, **imag_params)

# plt.subplot(122)

# plt.title('Interpolated')

# w /= np.linalg.norm(w, axis=1)[:, np.newaxis]

# plt.imshow(np.abs(w) ** 2, **imag_params)

for x_ in np.array(x_trajectory).T:
    plt.plot(x_, t_trajectory, 'r', alpha=1)

plt.show()
