
import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial
from jax.experimental.ode import odeint
import jaxopt

jax.config.update('jax_enable_x64', True)

from typing import Dict


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



@partial(jit, static_argnums=(0, 1, ) )
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

    n_steps = U.shape[0]

    def body(carry, u):
        t, x_prev = carry

        x = f(x_prev, u, t, theta)
        y = g(x_prev, u, t, theta)

        carry = (t + dt, x)

        return carry, ( t, x, y )

    carry, (T, X, Y) = lax.scan(body, (0.0, x0), U )
    
    X = jnp.vstack( (x0, X[:-1] ) )
        
    return T, X, Y



def _repeat_vec( w : jnp.ndarray, n : int ):
    
    assert len(w.shape) == 1, 'w must be a one-dimensional vector'
    return jnp.repeat( w.reshape( 1, w.shape[0] ), n, axis=0 )


def make_const_weights(wx : jnp.ndarray, wy : jnp.ndarray, n : int):
    """
        Expand a set of constant weights wx for the state residuals and weights 
        wy for the the outputs for use with estimate() .
    """
    
    Wx = _repeat_vec( wx, n-1 )
    Wy = _repeat_vec( wy, n )
    
    return Wx, Wy

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

def eval_e_X(f, X, U, T, theta):
    """
        compute the residuals (errors) for the state trajectory
    """

    # vectorize transition function f(x, u, t, theta)
    f_vec = vectorize_f(f)

    # step forward through transition function x_next( i ) = f( x(i), u(i), t(i), theta ) for all i
    X_next = f_vec( X, U, T, theta )

    # comput e_X( i ) = x( i+1 ) - x_next( i ) for all i
    e_X = X[1:] - X_next[:-1]

    return e_X

def eval_e_Y(g, X, U, T, theta, Y):
    """
        compute the residuals (errors) for the system output
    """
    
    # vectorize output function g(x, u, t, theta)
    g_vec = vectorize_g(g)

    # compute y_hat(i) = g( x(i), u(i), t(i), theta) for all i
    Y_hat = g_vec( X, U, T, theta )

    # residula compared to measued output sequence Y
    e_Y = Y_hat - Y
    
    return e_Y

def cost_fn(f, g, X, U, T, theta, Y, Wx, Wy):

    J = jnp.sum(
        (
            (Wx * eval_e_X(f, X, U, T, theta)).reshape(-1) 
        )**2
    ) + jnp.sum(
        (
            (Wy * eval_e_Y(g, X, U, T, theta, Y)).reshape(-1)
        )**2
    )
    
    return J

def objective( variables, parameters, static_parameters ):
    
    T, U, Y, Wx, Wy = parameters
    f, g            = static_parameters
    X, theta        = variables
    
    J = cost_fn(f, g, X, U, T, theta, Y, Wx, Wy)
    
    return J
    
    
@partial(jit, static_argnums=(0, 1, ) )
def estimate(f, g, T, U, Y, Wx, Wy, X0, theta0):
    """
        Estimation the state trajectory and the system parameters of a system from I/O-data
        
        The routine uses input-output data recorded from a system and a model 
        to estimate the state trajectory and a set of parameters.
        
        Args:
            f: the discrete-time system function with the prototype x_next = f(x, u, t, theta)
            g: the output function with the prototype y = g(x, u, t, theta)
            T: a time vector
            U: the input signal to the applied to the system
            Y: the output signal of the system in response to the input
            Wx: weight coefficients for the states residuals
            Wy: weight coefficients for the output residuals
            X0: an initial guess for the state trajectory
            theta0: an initial guess for the system parameters
        
        Returns: X_hat, theta_hat, Y_hat, J_star, res
            X_hat: the estimate state trajectory
            theta_hat: the estimated parameters
            Y_hat: the output computed from the estimated state trajectory and the identified parameters
            J_star: the final value of the cost function
            res: solver-internal information
            
    """
    
    # check for correct parameters
    assert len(T.shape) == 1
    assert len(X0.shape) == 2
    assert len(Y.shape) == 2
    
    n_steps = T.shape[0]
    n_states = X0.shape[1]
    n_outputs = Y.shape[1]
    
    assert U.shape[0] == n_steps
    assert U.shape[1] >= 1 # n_inputs
    
    assert Y.shape[0] == n_steps
    assert Y.shape[1] >= 1
    
    assert Wx.shape[0] == n_steps-1
    assert Wx.shape[1] == n_states
    
    assert Wy.shape[0] == n_steps
    assert Wy.shape[1] == n_outputs
    
    assert X0.shape[0] == n_steps
    
    # pack parameters and variables
    parameters = (T, U, Y, Wx, Wy)
    static_parameters = (f, g)
    variables = (X0, theta0)

    # pass static parameters into objective function
    objective_ = partial(objective, static_parameters=static_parameters)

    # run optimization
#    gd = jaxopt.GradientDescent(fun=objective_, maxiter=500)
    gd = jaxopt.BFGS(fun=objective_, value_and_grad=False, tol=0.000001, maxiter=5000)   
    
    res = gd.run(
        variables, 
        parameters=parameters, 
    )
    
    variables_star = res.params
    
    # unpack results
    X_hat, theta_hat = variables_star
    
    # compute the optimal cost
    J_star = cost_fn(f, g, X_hat, U, T, theta_hat, Y, Wx, Wy)
    
    # compute the output from the estimated states and parameters
    Y_hat = vectorize_g(g)(X_hat, U, T, theta_hat)
    
    return X_hat, theta_hat, Y_hat, J_star, res



