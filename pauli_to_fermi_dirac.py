"""
The Pauli master equation (classical continuous Markov process)
with the Fermi-Dirac distribution as the stationary state
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

###############################################################################
#
#   Set parameters
#
###############################################################################
# Randomly select energy levels
E = np.sort(np.random.rand(5))

# 1/kT
beta = 100.

# chemical potential
mu = np.random.uniform(E.min(), E.max())

# Randomly generate the rate coefficients (gamma_{n to j}) (n > j)
# of course in real application that need to be calculated from SH
gamma = np.random.rand(E.size, E.size)
gamma[np.tril_indices_from(gamma, -1)] = 0.

###############################################################################
#
#   Propagate the pauli equation
#
###############################################################################

# Set the remaining rate coefficients according to the thermalization condition
gamma += gamma.T * (np.exp(beta * (E[np.newaxis, :] - mu)) + 1.) \
            / (np.exp(beta * (E[:, np.newaxis] - mu)) + 1.)

def pauli_RHS(p, t, gamma):
    """
    The RHS of the Pauli master equation
    """
    return gamma.dot(p) - gamma.sum(axis=0) * p


# time range to be studied
t = np.linspace(0, 10, 1000)

# the initial condition
p0 = np.zeros(E.size)
p0[-1] = 1.

pop = odeint(pauli_RHS, p0, t, args=(gamma,))
pop = pop.T

# consistency checks
assert np.allclose(pop.sum(axis=0), 1., atol=1e-7), "The population must sum up to 1"
assert np.all(pop >= 0.), "The population must be nonegative"

###############################################################################
#
#   Plot the results of simulations
#
###############################################################################

# calculate the Dirac Fermi distribution
dirac_fermi = 1. / (np.exp(beta * (E - mu)) + 1.)
dirac_fermi /= dirac_fermi.sum()

plt.title("Pauli master equation with the Fermi-Dirac distribution as the stationary state")

for p in dirac_fermi:
    plt.plot([t[-1]], [p], '+', label='Dirac-Fermi')

for num, p in enumerate(pop):
    plt.plot(t, p, label="level " + str(num))

plt.xlabel("time")
plt.ylabel('population')
plt.legend()
plt.show()