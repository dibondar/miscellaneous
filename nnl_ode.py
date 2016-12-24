import numpy as np
from scipy.optimize import nnls
from scipy.linalg import expm, norm

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

    def integrate(self, t_final, atol=1e-18, rtol=1e-10):
        """
        Adaptively integrate the system of equations assuming self.t and self.y set the initial value.
        :param t_final: (scalar) the final time to be reached.
        :param atol: the absolute tolerance parameter
        :param rtol: the relative tolerance parameter
        :return: current value of y
        """
        if not np.isscalar(t_final):
            raise ValueError("t_final must be a scalar. If t_final is iterable consider using "
                             "the list comprehension [integrate(t) for t in times].")

        sign_dt = np.sign(t_final - self.t)

        # Loop util the final time moment is reached
        while sign_dt * (t_final - self.t) > 0:
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

            # Initial guess for the time-step
            dt = 0.25 / norm(self.M(self.t, *self.M_args, **self.M_kwargs))

            # time step must not take as above t_final
            dt = sign_dt * min(dt, abs(t_final - self.t))

            # Loop until optimal value of dt is not found (adaptive step size integrator)
            while True:
                M = self.M(self.t + 0.5 * dt, *self.M_args, **self.M_kwargs)
                M = np.array(M, copy=False)
                M *= -dt

                new_y, residual = nnls(expm(M), self.y)

                # Adaptive step termination criterion
                if np.allclose(residual, 0., rtol, atol):
                    # residual is small it seems we got the solution

                    # Additional check: If M is a transition rate matrix,
                    # then the sum of y must be preserved
                    if np.allclose(M.sum(axis=0), 0., rtol, atol):

                        # exit only if sum( y(t+dt) ) = sum( y(t) )
                        if np.allclose(sum(self.y), sum(new_y), rtol, atol):
                            break
                    else:
                        # M is not a transition rate matrix, thus exist
                        break

                if np.allclose(dt, 0., rtol, atol):
                    # print waring if dt is very small
                    print(
                        "Warning in nnl_ode: adaptive time-step became very small." \
                        "The numerical result may not be trustworthy."
                    )
                    break
                else:
                    # half the time-step
                    dt *= 0.5

            # the dt propagation is successfully completed
            self.t += dt
            self.y = new_y

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

    # randomly generate test parameters
    dump, pump = np.random.uniform(5, 20, 2)

    t = np.linspace(0., 1., 100)

    # Exact solutions
    p0 = (dump + pump * np.exp(-(dump + pump) * t)) / (dump + pump)
    p1 = pump * (1. - np.exp(-(dump + pump) * t)) / (dump + pump)

    # Numerical solutions
    def M(t, dump, pump):
        return np.array([[-pump, dump], [pump, -dump]])

    solver = nnl_ode(M, M_args=(dump, pump)).set_initial_value([1., 0.])

    # numerically propagate using nnl_ode
    p0_numeric, p1_numeric = np.array(
        [solver.integrate(tau) for tau in t]
    ).T

    print "\nNumerical error in p0 = %1.2e" % norm(p0 - p0_numeric)
    print "Numerical error in p1 = %1.2e\n" % norm(p1 - p1_numeric)

    # Plot
    plt.subplot(121)
    plt.plot(t, p0, 'r', label='exact')
    plt.plot(t, p0_numeric, 'b', label='nnl_ode')
    plt.xlabel('time')
    plt.ylabel('p0')
    plt.legend()

    plt.subplot(122)
    plt.plot(t, p1, 'r', label='exact')
    plt.plot(t, p1_numeric, 'b', label='nnl_ode')
    plt.xlabel('time')
    plt.ylabel('p1')
    plt.legend()

    plt.show()
