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




def vectorize_g(g):
    """ 
        vectorize the output function g(x, u, t, theta)
    """
    return jax.vmap( g, in_axes=(0, 0, 0, None) )

def vectorize_f(f):
    """ 
        vectorize the output function g(x, u, t, theta)
    """
    return jax.vmap( f, in_axes=(0, 0, 0, None) )


def eval_X_next(f, X, U, T, theta):

    # vectorize the transition function f(x, u, t, theta)
    f_vec = vectorize_f(f)

    # step forward through transition function x_next( i ) = f( x(i), u(i), t(i), theta ) for all i
    X_next = f_vec( X, U, T, theta )

    return X_next

