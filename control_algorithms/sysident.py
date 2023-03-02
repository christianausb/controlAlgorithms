
import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial
from jax.experimental.ode import odeint
import jaxopt

jax.config.update('jax_enable_x64', True)

from typing import Dict

#
# simulation
#

def vectorize_output_function(output_function, theta):
    """
        helper function to derive a vectorized form of a given
        model output function 
    """

    v_output_function = jax.vmap( 
        partial( output_function, theta=theta ), 
        in_axes=(0, 0) 
    )
    
    return v_output_function


def sample_and_hold( U : jnp.ndarray, dt, t ):
    """
        Model a continuous signal by applying sample and hold
        at equidistant samples given in the array U. Then, sample
        this continuous signal at t.
    """

    k = t // dt
    k = jnp.array(k, dtype=jnp.int32)
    k = jnp.clip( k, 0, U.shape[0]-1 )

    return U[k], k
    
def dynamics_input_sampler( state, t, theta, dynamics, U : jnp.ndarray, dt_U):
    """
        applies the given input signal to the given dynamic via an external input u
    """
    
    # use sample and hold method to derive a continuous signal from the given samples
    u, _ = sample_and_hold(U, dt_U, t)
    
    # execute the wrapped dynamics function
    return dynamics( state, t, theta, u )
    

def simulate(dynamics, output_function, theta, X0, pars, n=100):
    """
        Simulate a model given its dynamics and output function

        Args:
            dynamics: a function f(state, t, theta, u) implementing the system function f
            output_function: a function g(state, t, theta) that implements the system output function g
            theta: the parameters of the system (as a JAX pytree)
            x0: a vector describing the initial states.
            pars: dictionary as returned by make_ident_pars(), only the system input and the output sampling 
                  time dt_Y needs to be specified therein.
            n: number of samples to simulate


        Returns:
            a tuple describing
                 - T: the time vector
                 - X: the state trajectory
                 - Y: the system output trace
    """
    
    T = jnp.arange(n) * pars['dt_Y']
    output_fn = vectorize_output_function(output_function, theta=theta)
    
    dynamics_with_sample_and_hold_input = partial(
        dynamics_input_sampler, 
        dynamics=dynamics, 
        U=pars['U_excitation'], dt_U=pars['dt_U'],
    )
    
    X = odeint( 
        partial( dynamics_with_sample_and_hold_input, theta=theta ), 
        X0,
        T,
    )

    Y = output_fn(X, T) 
        
    return T, X, Y

#   
# sysid
#


def evaluate(
    dynamics,                # static
    output_function,         # static
    optim_variables,         # dyn
    pars,                    # dyn
    
):
    """
        Evaluate the simulation results and compute a cost (loss)
    """
    
    Y_measurement = pars['Y_measurement']
    dt = pars['dt_Y']
    
    # unpack variables
    theta, x0 = optim_variables
    
    T, X, y = simulate(dynamics, output_function, theta, x0, pars, Y_measurement.shape[0])
    
    # eval cost (2-norm)
    J =  jnp.sum( (y - Y_measurement) ** 2 )
    
    return T, X, y, J

def objective(dynamics, output_function, optim_variables, pars):
    T, X, y, J = evaluate(dynamics, output_function, optim_variables, pars)
    return J
    

def compute_gradient(dynamics, output_function, optim_variables, pars):
    """
        Compute the gradient of the cost
    """
        
    def fun(optim_variables, pars):
        return objective(dynamics, output_function, optim_variables, pars)
        
    J, grad = jax.value_and_grad(fun)( optim_variables, pars )
        
    return J, grad


    
partial(jit, static_argnums=(0, 1, ) )
def run_jaxopt_opt(
    dynamics, 
    output_function, 
    optim_variables0,
    pars,    
):
    print('jit compile')

    #
    optim_variables = optim_variables0
    
    def fun( optim_variables, pars, dynamics, output_function ):        
        return objective(dynamics, output_function, optim_variables, pars)

    fun_ = partial(fun, dynamics=dynamics, output_function=output_function)
    
#    gd = jaxopt.GradientDescent(fun=fun_, value_and_grad=False, maxiter=5090, implicit_diff=True)
    gd = jaxopt.BFGS(fun=fun_, value_and_grad=False)
    res = gd.run(optim_variables, pars=pars)    
    optim_variables_star = res.params
    
    return optim_variables_star, res
    
    

#
# ident interface
#
    

def make_ident_pars(
    Y_measurement : jnp.ndarray = None,
    dt_Y : float = None,
    U_excitation : jnp.ndarray = None,
    dt_U : float = None,
) -> Dict:
    """
        Combine I/O data for the identification procedure

        Args:
            Y_measurement: measured output data of the system to identify
            dt_Y: the sampling time of the data Y_measurement
            U_excitation: the system input signal used to excite the system
            dt_Y: the sampling time of the data U_excitation

        Returns:
            A dictionary to be used by the routine identify()
    """
    pars = {
        'Y_measurement' : Y_measurement,
        'dt_Y' : dt_Y,
        'U_excitation' : U_excitation,
        'dt_U' : dt_U,
    }
    return pars


def identify(
    dynamics,
    output_function,
    theta_guess, 
    x0, 
    pars,
    method : str = 'jaxopt',
):
    """
        Perform system identification given a prototype model and measured data

        Args:
            dynamics: a function f(state, t, theta, u) implementing the system function f
            output_function: a function g(state, t, theta) that implements the system output function g
            theta_guess: an initial guess of the parameters to identify (as a JAX pytree)
            x0: a vector describing a initial guess for the initial states to be identified.
            pars: dictionary as returned by make_ident_pars()
            method: the method to apply (default: 'jaxopt')

        Returns:
            A tuple containing 
                - theta_found: the identified parameters (as a JAX pytree, like theta_guess)
                - x0: a vector containing the identified inital states
                - res: internal information as returned from the optimizer
    """
    
    optim_variables_0 = ( theta_guess, x0, )
    
    if method.lower() == 'jaxopt':
        optim_variables_star, res = run_jaxopt_opt(
            dynamics, 
            output_function, 
            optim_variables_0,
            pars,    
        )

    else:
        raise BaseException('Method ' + method + ' is not supported')
    
    theta_found = optim_variables_star[0]
    x0_found    = optim_variables_star[1]
    
    return theta_found, x0_found, res
    
    