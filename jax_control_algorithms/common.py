import jax
import jax.numpy as jnp
from jax import jit
from jax import lax

from functools import partial
import math


def euler(f, dt):
    """
        apply Euler forward integrator

        Args:
            f: the transition function of the continuous system to which to apply the integration
            dt: fixed sampling time to be used in the integration

        Returns:
            a new transition function that describes the discrete-time system resulting after application
            of the integration method 
    """
    return lambda x, u, t, theta: x + dt * f(x, u, t, theta)


def rk4(f, dt):
    """
        apply Runge Kutta 4-th order integrator

        Args:
            f: the transition function of the continuous system to which to apply the integration
            dt: fixed sampling time to be used in the integration

        Returns:
            a new transition function that describes the discrete-time system resulting after application
            of the integration method 
    """

    def integrator(x, u, t, theta):

        dt2 = dt / 2.0
        k1 = f(x, u, t, theta)
        k2 = f(x + dt2 * k1, u, t, theta)
        k3 = f(x + dt2 * k2, u, t, theta)
        k4 = f(x + dt * k3, u, t, theta)

        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return integrator

@partial(jit, static_argnums=(
    0,
    1,
))
def simulate_dscr(f, g, x0, U, dt, theta):
    """
        Perform a discrete-time simulation of a system
        
        Args:
            f: the discrete-time system function with the prototype x_next = f(x, u, t, theta)
            g: the output function with the prototype y = g(x, u, t, theta)
            X0: the initial state of the system
            U: the input signal to the applied to the system
            dt: the sampling time (used to generate the time vector T)
            theta: the system parameters
        
        Returns: T, X, Y
            T: a time vector
            X: the state trajectory
            Y: the output signal of the system in response to the input
    """

    def body(carry, u):
        t, x_prev = carry

        x = f(x_prev, u, t, theta)
        y = g(x_prev, u, t, theta)

        carry = (t + dt, x)

        return carry, (t, x, y)

    carry, (T, X, Y) = lax.scan(body, (0.0, x0), U)

    X = jax.tree_util.tree_map(lambda x0, X: jnp.vstack((x0, X[:-1])), x0, X)

    return T, X, Y


def vectorize_g(g):
    """ 
        vectorize the output function g(x, u, t, theta)

        Allow the vectorized evaluation of the function g

        Returns:
            the vectorized function of g
    """
    return jax.vmap(g, in_axes=(0, 0, 0, None))


def vectorize_f(f):
    """ 
        vectorize the output function g(x, u, t, theta)

        Allow the vectorized evaluation of the function f

        Returns:
            the vectorized function of f
    """
    return jax.vmap(f, in_axes=(0, 0, 0, None))


def eval_X_next(f, X, U, T, theta):
    """
        Evaluate the one-step ahead prediction

        For each sample of the given state trajectory X that next state is predicted 
        by evaluating the transition function f. Herein, the respective control inputs
        U are applied.

            X_next( i ) = f( X(i), U(i), T(i), theta ) for all i

        Args:
            f: the discrete-time transition function of the system
            X: the state trajectory
            U: the trajectory of control inputs
            T: the sequence of time instances
            theta: parameters to the function f
        Returns:
            X_nest: the sequence of one-step ahead predictions
    """

    # vectorize the transition function f(x, u, t, theta)
    f_vec = vectorize_f(f)

    # step forward through transition function x_next( i ) = f( x(i), u(i), t(i), theta ) for all i
    X_next = f_vec(X, U, T, theta)

    return X_next
