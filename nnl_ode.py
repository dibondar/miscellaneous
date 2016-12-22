import numpy as np
from scipy.optimize import nnls
from scipy.linalg import expm

__doc__ = """
Non-negative solver for a system of linear ordinary differential equations.

Note that scipy lacks such an integrator (as of the end of 2016)
"""

class nnl_ode:
    """
    Solve an equation system :math:`y'(t) = M(t)y` where y(t) is non-negative and M(t) is a matrix.
    """
    def __init__(self, M, M_args=(), M_kwargs={}):
        """

        :param M: callable ``M(t, *M_args, **M_kwargs)``
                The matrix in the Rhs of the equation. t is a scalar,
                ``M_args`` and ``M_kwargs``is set by calling ``set_M_params(*M_args, **M_kwargs)``.
                `M` should return an NxN container (e.g., array), where `N = len(y)` is the number of variables.

        :param M_params: (optional) parameters for user-supplied function ``M``
        """
        self.M = M
        self.set_M_params(*M_args, **M_kwargs)

    def set_M_params(self, *M_args, **M_kwargs):
        """
        Set extra parameters for user-supplied function M.
        """
        self.M_args = M_args
        self.M_kwargs = M_kwargs
        return self

    def set_initial_value(self, y, t=0.0):
        """Set initial conditions y(t) = y."""
        self.t = t
        self.y = y
        return self

    def integrate(self, t_final, dt=0.01):
        """
        Integrate the system of equations with fixed step size dt assuming self.t and self.y set the initial value.
        :param t_final: the final time to be reached.
        :param dt: step size
        :return: current value of y
        """
        # get number of steps to be integarted over
        n_steps = int(np.ceil((t_final - self.t) / dt))

        # update the step size such that self.t + dt * n_steps = t_final
        dt = ((t_final - self.t) / n_steps if n_steps else 0.)

        # Loop utill the final time moment is reached
        while self.t < t_final:
            #######################################################################################
            #
            #           Description of numerical methods
            #
            #   A formal solution of the system of linear ode y'(t) = M(t) y(t) reads as
            #
            #       y(t) = T exp[ \int_{t_init}^{t_fin} M(\tau) d\tau ] y(t_init)
            #
            #   where T exp is a Dyson time-ordered exponent. Hence,
            #
            #       y(t + dt) = T exp[ \int_{t}^{t+dt} M(\tau) d\tau ] y(t).
            #
            #   Dropping the time ordering operation leads to the cubic error
            #
            #       y(t + dt) = exp[ \int_{t}^{t+dt} M(\tau) d\tau ] y(t) + O( dt^3 ).
            #
            #   Employing the mid-point rule for the integration also leads to the cubic error
            #
            #       y(t + dt) = exp[  M(t + dt / 2) dt ] y(t) + O( dt^3 ).
            #
            #   Therefore, we finally get the linear equation w.r.t. unknown y(t + dt) [note y(t) is known]
            #
            #       exp[  -M(t + dt / 2) dt ] y(t + dt) = y(t) + O( dt^3 ),
            #
            #   which can be solved by scipy.optimize.nnls ensuring the non-negativity constrain for y(t + dt).
            #
            #######################################################################################
            M = self.M(self.t + 0.5 * dt, *self.M_args, **self.M_kwargs)
            M = np.array(M, copy=False)
            M *= -dt

            print np.linalg.eigvals(M)

            self.y, rnorm = nnls(expm(M), self.y)

            #print rnorm

            self.t += dt

        return self.y

###################################################################################################
#
#   Test
#
###################################################################################################

if __name__=='__main__':

    import matplotlib.pyplot as plt

    ################################################################################################
    #
    #   Solve the system
    #
    #       p0'(t) = -pump * p0(t) + dump * p1(t), p1'(t) = pump * p0(t) - dump * p1(t)
    #
    #   with the initial condition p0(0) = 1, p1(0) = 0
    #
    ################################################################################################

    #np.random.seed(38490)

    # randomly generate test parameters
    dump, pump = np.random.uniform(0, 1, 2)



    # time grid
    t = np.linspace(0, 5, 100)
    dt = t[1] - t[0]

    # Exact solutions
    p0 = (dump + pump * np.exp(-(dump + pump) * t)) / (dump + pump)
    p1 = pump * (1. - np.exp(-(dump + pump) * t)) / (dump + pump)

    # Numerical solutions
    def M(t, dump, pump):
        return [[-pump, dump], [pump, -dump]]

    solver = nnl_ode(M, M_args=(dump, dump)).set_initial_value([1., 0.])

    # numerically propagate
    p0_numeric, p1_numeric = np.array(
        [solver.integrate(tau, dt=0.001) for tau in t]
    ).T

    print np.linalg.norm(
        np.gradient(p0, dt) - (-pump * p0 + dump * p1),
        np.inf
    )
    print np.linalg.norm(
        np.gradient(p1, dt) - (pump * p0 - dump * p1),
        np.inf
    )

    print np.linalg.norm(
        np.gradient(p0_numeric, dt) - (-pump * p0_numeric + dump * p1_numeric),
        np.inf
    )
    print np.linalg.norm(
        np.gradient(p1_numeric, dt) - (pump * p0_numeric - dump * p1_numeric),
        np.inf
    )

    plt.subplot(121)
    plt.plot(t, p0, 'r', label='exact')
    plt.plot(t, p0_numeric, 'b', label='numeric')
    plt.xlabel('time')
    plt.ylabel('p0')
    plt.legend()

    plt.subplot(122)
    plt.plot(t, p1, 'r', label='exact')
    plt.plot(t, p1_numeric, 'b', label='numeric')
    plt.xlabel('time')
    plt.ylabel('p1')
    plt.legend()

    plt.show()
