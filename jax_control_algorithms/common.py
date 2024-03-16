import jax
import jax.numpy as jnp
from jax import jit
from jax import lax

from functools import partial
import math


def euler(f, dt):
    return lambda x, u, t, theta: x + dt * f(x, u, t, theta)


def rk4(f, dt):

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

    # n_steps = U.shape[0]

    def body(carry, u):
        t, x_prev = carry

        x = f(x_prev, u, t, theta)
        y = g(x_prev, u, t, theta)

        carry = (t + dt, x)

        return carry, (t, x, y)

    carry, (T, X, Y) = lax.scan(body, (0.0, x0), U)

#    X = jnp.vstack((x0, X[:-1]))
    X = jax.tree_util.tree_map(
        lambda x0, X : jnp.vstack((x0, X[:-1])),
        x0, X
    )

    return T, X, Y


def vectorize_g(g):
    """ 
        vectorize the output function g(x, u, t, theta)
    """
    return jax.vmap(g, in_axes=(0, 0, 0, None))


def vectorize_f(f):
    """ 
        vectorize the output function g(x, u, t, theta)
    """
    return jax.vmap(f, in_axes=(0, 0, 0, None))


def eval_X_next(f, X, U, T, theta):

    # vectorize the transition function f(x, u, t, theta)
    f_vec = vectorize_f(f)

    # step forward through transition function x_next( i ) = f( x(i), u(i), t(i), theta ) for all i
    X_next = f_vec(X, U, T, theta)

    return X_next
