import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
from jax.experimental.ode import odeint
import jaxopt
from functools import partial

import math
from typing import Dict

from jax_control_algorithms.common import *



def get_n_steps(max_time, dt):
    n_steps = int(round( max_time / dt ))
    return n_steps

def make_exp_decay_weights(lam, n_steps, dtype=jnp.float64):
    return jnp.exp( -jnp.linspace(0, 1.0, n_steps, dtype=dtype) * lam ).reshape( (n_steps,1) )

def eval_R_squared( y, y_hat, weights=None ):
    """
        Compute R^2 metric describing how well a model fits the given data

        https://en.wikipedia.org/wiki/Coefficient_of_determination
    """

    assert len(y.shape) == 1
    assert len(y_hat.shape) == 1
    assert y.shape[0] == y_hat.shape[0]
    
    if weights is not None:

        assert len(weights.shape) == 1
        assert y.shape[0] == weights.shape[0]
        SS_res = jnp.mean( weights * (y - y_hat)**2 ) / jnp.mean(weights)

    else:
        SS_res = jnp.mean( (y - y_hat)**2 )


    SS_tot = jnp.clip( 
        jnp.mean( (y - jnp.mean(y))**2 ),
        0.001, None
    )

    R_sq_ = 1 - (SS_res / SS_tot)
    dB_R_sq_ = 20 * jnp.log10( 
        jnp.clip( R_sq_, 0.0001, None )  # saturate at small values close to zero (prevent also negative metrics)
    )
    
    return R_sq_, dB_R_sq_
    


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


def eval_e_X(X, X_next):
    """
        compute the residuals (errors) for the state trajectory
    """

    # compute e_X( i ) = x( i+1 ) - x_next( i ) for all i
    e_X = X[1:] - X_next[:-1]

    return e_X


def eval_Y_hat(g, X, U, T, theta, Y):
    # vectorize output function g(x, u, t, theta)
    g_vec = vectorize_g(g)

    # compute y_hat(i) = g( x(i), u(i), t(i), theta) for all i
    Y_hat = g_vec( X, U, T, theta )

    return Y_hat

def eval_e_Y(Y, Y_hat):
    """
        compute the residuals (errors) for the system output
    """
    
    # residuals compared to measured output sequence Y
    e_Y = Y_hat - Y
    
    return e_Y


#
# routine for state estimation and parameter identification
#

def cost_fn(f, g, X, U, T, theta, Y, Wx, Wy):

    X_next = eval_X_next(f, X, U, T, theta)
    Y_hat  = eval_Y_hat(g, X, U, T, theta, Y)

    J = jnp.mean(
        (
            Wx * eval_e_X(X, X_next)**2 
        ).reshape(-1)
    ) + jnp.mean(
        (
            Wy * eval_e_Y(Y, Y_hat)**2
        ).reshape(-1)
    )
    
    return J, X_next, Y_hat

def __objective( variables, parameters, static_parameters ):
    
    T, U, Y, Wx, Wy = parameters
    f, g            = static_parameters
    X, theta        = variables
    
    J, X_next, Y_hat = cost_fn(f, g, X, U, T, theta, Y, Wx, Wy)
    
    return J, X_next, Y_hat
    
def objective( variables, parameters, static_parameters ):
    return __objective( variables, parameters, static_parameters )[0]


@partial(jit, static_argnums=(0, 1, ) )
def estimate_objective(f, g, T, U, Y, Wx, Wy, X, theta):

    # pack parameters and variables
    parameters = (T, U, Y, Wx, Wy)
    static_parameters = (f, g)
    variables = (X, theta)

    # pass static parameters into objective function
    objective_ = partial(__objective, static_parameters=static_parameters)

    #
    J, X_next, Y_hat = objective_( variables, parameters )

    return J, X_next, Y_hat



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
            U: the vectorial input signal to the applied to the system
            Y: the vectorial output signal of the system in response to the input
            Wx: weight coefficients for the states residuals
            Wy: weight coefficients for the output residuals
            X0: an initial guess for the state trajectory
            theta0: an initial guess for the system parameters
        
        Returns: X_hat, theta_hat, Y_hat, J_star, res
            X_hat: the estimate state trajectory
            theta_hat: the estimated parameters
            Y_hat: the vectorial output signal computed from the estimated state trajectory 
                   and the identified parameters
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
    J_star, _, Y_hat = cost_fn(f, g, X_hat, U, T, theta_hat, Y, Wx, Wy)
        
    return X_hat, theta_hat, Y_hat, J_star, res


#
# routine for identification (without state estimation)
#


def cost_fn_identification(f, g, X, U, T, theta, Y, Wy):

    J = jnp.mean(
        (
            Wy * eval_e_Y(g, X, U, T, theta, Y)**2
        ).reshape(-1)
    )
    
    return J

def objective_identification( variables, parameters, static_parameters ):
    
    T, U, Y, Wy     = parameters
    f, g            = static_parameters
    theta, x0       = variables
    
    # simulate the system 
    _, X, Y_hat = simulate_dscr(f, g, x0, U, 1.0, theta)
    
    # eval 
    J = cost_fn_identification(f, g, X, U, T, theta, Y, Wy)
    
    return J
    

@partial(jit, static_argnums=(0, 1, ) )
def identify(f, g, T, U, Y, Wy, x0, theta0):
    """
        Indentify the system parameters and the initial state of a system from I/O-data
        
        The routine uses input-output data recorded from a system and a model 
        to estimate.
        
        Args:
            f: the discrete-time system function with the prototype x_next = f(x, u, t, theta)
            g: the output function with the prototype y = g(x, u, t, theta)
            T: a time vector
            U: the vectorial input signal to the applied to the system
            Y: the vectorial output signal of the system in response to the input
            Wy: weight coefficients for the output residuals
            x0: an initial guess for the initial state of the system
            theta0: an initial guess for the system parameters
        
        Returns: X_hat, theta_hat, Y_hat, J_star, res
            X_hat: the estimate state trajectory
            theta_hat: the estimated parameters
            Y_hat: the vectorial output signal computed from the estimated state trajectory 
                   and the identified parameters
            J_star: the final value of the cost function
            res: solver-internal information
            
    """
    
    # check for correct parameters
    assert len(T.shape) == 1
    assert len(Y.shape) == 2
    
    n_steps = T.shape[0]
    n_outputs = Y.shape[1]
    
    assert U.shape[0] == n_steps
    assert U.shape[1] >= 1 # n_inputs
    
    assert Y.shape[0] == n_steps
    assert Y.shape[1] >= 1
        
    assert Wy.shape[0] == n_steps
    assert Wy.shape[1] == n_outputs
    
#    assert len(x0.shape[0] == n_steps
    
    # pack parameters and variables
    parameters = (T, U, Y, Wy)
    static_parameters = (f, g)
    variables = (theta0, x0)

    # pass static parameters into objective function
    objective_ = partial(objective_identification, static_parameters=static_parameters)

    # run optimization
#    gd = jaxopt.GradientDescent(fun=objective_, maxiter=500)
    gd = jaxopt.BFGS(fun=objective_, value_and_grad=False, tol=0.000001, maxiter=5000)   
    
    res = gd.run(
        variables, 
        parameters=parameters, 
    )
    
    variables_star = res.params
    
    # unpack results
    theta_hat, x0_hat = variables_star
    
    # simulate with found parameters
    _, X_hat, Y_hat = simulate_dscr(f, g, x0_hat, U, 1.0, theta_hat)

    # compute the optimal cost
    J_star = cost_fn_identification(f, g, X_hat, U, T, theta_hat, Y, Wy)
        
    return theta_hat, x0_hat, X_hat, Y_hat, J_star, res


