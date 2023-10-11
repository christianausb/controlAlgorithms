import jax
from jax import jit
from jax import lax
from jax import vmap
import jax.numpy as jnp
import jaxopt

from functools import partial
from inspect import signature
from dataclasses import dataclass

import math
from jax_control_algorithms.common import *

import time

#jax.config.update('jax_enable_x64', True)




def constraint_geq(x, v):
    """
        x >= v
    """
    return x - v

def constraint_leq(x, v):
    """
        x <= v
    """
    return v - x


def boundary_fn(x, t_opt, y_max = 10, is_continue_linear=False):
    
    # assert y_max > 0
    
    # which x yields -1/t_opt * log(x) = y_max
    # exp(log(x)) = exp( -y_max * t_opt )
    # AW: x_thr = exp( -y_max * t_opt )
    
    x_thr = jnp.exp( -y_max * t_opt )

    # what is d/dx (-1/t_opt) * jnp.log(x) with x=x_thr ?
    # AW: (-1/t_opt) * 1/x_thr
    
    ddx = (-1/t_opt) * 1/x_thr
    
    # linear continuation for x < x_thr (left side)
    if is_continue_linear:
        _ddx = jnp.clip( ddx, -y_max*10, 0 )
        x_linear_cont = _ddx * (x - x_thr) + y_max
    else:
        x_linear_cont = y_max
    
    x_boundary_fn = - (1/t_opt) * jnp.log(x) 
    
    #
    y = jnp.where(
        x < x_thr, 
        x_linear_cont, 
        x_boundary_fn
    )
    
    return y



def print_if_nonfinite(text : str, x):
    is_finite = jnp.isfinite(x).all()
    
    def true_fn(x):
        pass
    def false_fn(x):
        # jax.debug.breakpoint()
        jax.debug.print(text, x=x)

    lax.cond(is_finite, true_fn, false_fn, x)
    
def print_if_outofbounds(text : str, x, x_min, x_max, var_to_also_print=None):
    is_oob = jnp.logical_and(
        jnp.all( x > x_min ),
        jnp.all( x < x_max ),        
    )
    
    def true_fn(x):
        pass
    def false_fn(x):
        # jax.debug.breakpoint()
        jax.debug.print(text, x=x)
        if var_to_also_print is not None:
            jax.debug.print('var_to_also_print={x}', x=var_to_also_print)

    lax.cond(is_oob, true_fn, false_fn, x)
    
     


#
# routine for state estimation and parameter identification
#

def eq_constraint(f, terminal_state_eq_constraints, X_opt_var, U_opt_var, K, x0, theta, power):
    """
    algebraic constraints for the system dynamics
    """

    X = jnp.vstack(( x0 , X_opt_var ))
    
    X_next = eval_X_next(f, X[:-1], U_opt_var, K, theta)

    # compute c_eq( i ) = x( i+1 ) - x_next( i ) for all i
    c_eq_running =  jnp.exp2(power) * X[1:] -  jnp.exp2(power) * X_next

    if terminal_state_eq_constraints is not None:
        # terminal constraints are defined
        x_terminal = X_opt_var[-1]
        
        
        number_parameters_to_terminal_fn =len( signature( terminal_state_eq_constraints ).parameters )
        if number_parameters_to_terminal_fn == 2:
            # the constraint function implements the power parameter
            
            c_eq_terminal = jnp.exp2(power) * terminal_state_eq_constraints(x_terminal, theta)
            
        elif number_parameters_to_terminal_fn == 3:
            
            c_eq_terminal = terminal_state_eq_constraints(x_terminal, theta, power)
        
        
        # total
        c_eq = jnp.vstack( (c_eq_running, c_eq_terminal) )
    else:
        # no terminal constraints are considered
        c_eq = c_eq_running

    return c_eq
    
def vectorize_running_cost(f_rk):
    """ 
        vectorize the running cost function running_cost(x, u, t, theta)
    """
    return jax.vmap( f_rk, in_axes=(0, 0, 0, None) )
  

def cost_fn(f, running_cost, X_opt_var, U_opt_var, K, theta):
 
    # cost
    J_trajectory = vectorize_running_cost(running_cost)(X_opt_var, U_opt_var, K, theta)

    J = jnp.mean(J_trajectory)
    return J

def __objective_penality_method( variables, parameters, static_parameters ):
    
    K, theta, x0, opt_t, opt_c_eq                      = parameters
    f, terminal_state_eq_constraints, inequ_constraints, running_cost = static_parameters
    X, U                                                              = variables
    
    n_steps = X.shape[0]
    assert U.shape[0] == n_steps
    
    #

    OPTION = 'A'

    if OPTION in ['A', 'B']:
        power = 0
    elif OPTION in ['C', 'D']:
        power = 10

    # get equality constraint. The constraints are fulfilled of all elements of c_eq are zero
    c_eq = eq_constraint(f, terminal_state_eq_constraints, X, U, K, x0, theta, power).reshape(-1)
    c_ineq = inequ_constraints(X, U, K, theta).reshape(-1)        

    # equality constraints using penality method
    
    if OPTION == 'A':
        J1_A = opt_c_eq * jnp.mean(
            ( c_eq.reshape(-1) )**2
        )
        J_equality_costs = J1_A
    
    if OPTION == 'B':
        opt_t_sqrt = jnp.sqrt(opt_t)
        J1_B = opt_c_eq * jnp.mean(
            ( opt_t_sqrt * c_eq.reshape(-1) )**2
        )
        J_equality_costs = J1_B
    
    if OPTION == 'C':
        J1_C = opt_c_eq/jnp.exp2(power)**2 * opt_t * jnp.mean(
            ( c_eq.reshape(-1) )**2
        )
        J_equality_costs = J1_C
    
    if OPTION == 'D':
        a = c_eq.reshape(-1)
        J1_D = opt_c_eq/jnp.exp2(power)**2 * opt_t * jax.numpy.linalg.norm( a, 2 )**2 / a.shape[0]
        J_equality_costs = J1_D

#    print_if_outofbounds('WARN; J1_A - J1_B = {x}', J1_A - J1_B, -0.00001, 0.00001)
   # print_if_outofbounds('WARN: J1_A - J1_B = {x}', J1_A - J1_B,       -0.001, 0.001, {'opt_t': opt_t, 'opt_t_sqrt': opt_t_sqrt, 'c_eq': c_eq} )

    # CHOICE
    #
        
    J_cost_function = cost_fn(f, running_cost, X, U, K, theta)
    
    J_boundary_costs = jnp.mean(
        boundary_fn(c_ineq, opt_t, 11, True)
    )
    
    return J_equality_costs + J_cost_function + J_boundary_costs, c_eq


def __feasibility_metric_penality_method(variables, parameters, static_parameters ):
    
    K, theta, x0                                       = parameters
    f, terminal_state_eq_constraints, inequ_constraints, running_cost = static_parameters
    X, U                                                              = variables
    
    # get equality constraint. The constraints are fulfilled of all elements of c_eq are zero
    c_eq = eq_constraint(f, terminal_state_eq_constraints, X, U, K, x0, theta, 0)
    c_ineq = inequ_constraints(X, U, K, theta)
    
    #
    metric_c_eq   = jnp.max(  jnp.abs(c_eq) )
    metric_c_ineq = jnp.max( -jnp.where( c_ineq > 0, 0, c_ineq ) )
    
    return metric_c_eq, metric_c_ineq

def objective_penality_method( variables, parameters, static_parameters ):
    return __objective_penality_method( variables, parameters, static_parameters )[0]

def feasibility_metric_penality_method(variables, parameters, static_parameters ):
    return __feasibility_metric_penality_method(variables, parameters, static_parameters )



def _verify_step(verification_state, i, res_inner, variables, parameters, opt_t, feasibility_metric_fn, t_final, eq_tol, verbose : bool):
    
    trace, _, = verification_state

    #
    is_X_finite = jnp.isfinite(variables[0]).all()
    is_abort_because_of_nonfinite = jnp.logical_not(is_X_finite)

    # verify step
    max_eq_error, max_ineq_error = feasibility_metric_fn(variables, parameters)
    n_iter_inner = res_inner.state.iter_num
    
    # verify metrics and check for convergence
    neq_tol = 0.0001

    is_eq_converged  = max_eq_error < eq_tol
    is_neq_converged = max_ineq_error < neq_tol # check if the solution is inside (or close) to the boundary

    is_converged = jnp.logical_and(
        is_eq_converged,
        is_neq_converged,
    )
    
    # trace
    trace_next = ( 
        trace[0].at[i].set(max_eq_error),
        trace[1].at[i].set(max_ineq_error),
        trace[2].at[i].set(n_iter_inner),
    )

    # As being in the 2nd iteration, compare to prev. metric and see if it got smaller
    is_metric_check_active = i > 2

    def true_fn(par):
        (i, trace, ) = par

        delta_max_eq_error = trace[0][i] - trace[0][i-1]
        is_abort  = delta_max_eq_error >= 0

        return is_abort


    def false_fn(par):
        (i, trace, ) = par
        return False

    is_abort_because_of_metric = lax.cond(is_metric_check_active, true_fn, false_fn, ( i, trace, ) )
    i_best = None    

    is_abort = jnp.logical_or(
        is_abort_because_of_nonfinite,
        is_abort_because_of_metric
    )

    def my_round(x, decimals=2):
        scale = jnp.array(10.0**decimals, dtype=jnp.float64)
        return (
            jnp.array(jnp.round( scale * x, decimals=0 ), dtype=jnp.int32) / scale
        )
    
    def my_to_int(x):
        return jnp.array(x, dtype=jnp.int32)


    if verbose:
        
        jax.debug.print(
            "🔄 it={i} \t (sub iter={n_iter_inner})\tt/t_final = {opt_t} %\teq_error/eq_tol = {max_eq_error} %\tbounds ok: {is_neq_converged}",
            i=i,    opt_t  = my_to_int(my_round(100 * opt_t / t_final, decimals=0)),
            max_eq_error   = my_to_int(my_round(100 * max_eq_error / eq_tol , decimals=0)),
            n_iter_inner   = n_iter_inner,
            is_neq_converged = is_neq_converged,
        )
        
        if False:  # additional info (for debugging purposes)
            jax.debug.print(
                "   is_abort_because_of_nonfinite={is_abort_because_of_nonfinite} is_abort_because_of_metric={is_abort_because_of_metric}) " + 
                "is_eq_converged={is_eq_converged}, is_neq_converged={is_neq_converged}",
                is_abort_because_of_nonfinite=is_abort_because_of_nonfinite,
                is_abort_because_of_metric=is_abort_because_of_metric,
                is_eq_converged=is_eq_converged,
                is_neq_converged=is_neq_converged,
            )

    # verification_state, is_finished, is_abort, i_best            
    return ( trace, is_converged, ), is_converged, is_eq_converged, is_abort, is_X_finite, i_best

def _optimize_trajectory( 
        i, variables, parameters, opt_t, opt_c_eq, verification_state_init, lam,
        tol_inner, t_final, max_iter_boundary_method,
        max_iter_inner, objective_fn, verification_fn,
        verbose, print_errors, target_dtype
    ):

    # convert dtypes
    ( variables, parameters, opt_t, opt_c_eq, verification_state_init, lam, tol_inner, ) = _convert_dtype(
        ( variables, parameters, opt_t, opt_c_eq, verification_state_init, lam, tol_inner, ),
        target_dtype
    )

    #
    # loop:
    # opt_t_init -> opt_t, opt_t = opt_t * lam
    #
    
    def loop_body(loop_par):
            
        #
        parameters_ = loop_par['parameters'] + ( loop_par['opt_t'], loop_par['opt_c_eq'], )

        # run optimization
        gd = jaxopt.BFGS(fun=objective_fn, value_and_grad=False, tol=loop_par['tol_inner'], maxiter=max_iter_inner)
        res = gd.run(loop_par['variables'], parameters=parameters_)
        _variables_next = res.params

        # run callback
        verification_state_next, is_finished, is_eq_converged, is_abort, is_X_finite, i_best = verification_fn(
            loop_par['verification_state'], 
            loop_par['i'], 
            res, _variables_next, 
            loop_par['parameters'], 
            loop_par['opt_t']
        )

        # t-control , t_final -> t_final
        is_finished = jnp.logical_and(is_finished, loop_par['opt_t'] >= loop_par['t_final'])
        opt_t_next = jnp.clip(loop_par['opt_t'] * lam, 0, loop_par['t_final']) 

        # c_eq-control
        opt_c_eq_next = jnp.where(
            is_eq_converged,

            # in case of convergence of the error below the threshold there is not need to increase c_eq
            loop_par['opt_c_eq'],

            # increase c_eq
            loop_par['opt_c_eq'] * lam,
        )

        #
        variables_next = (
            jnp.where(
                is_abort, 
                loop_par['variables'][0],      # use previous state of the iteration in case of abortion
                _variables_next[0] #
            ),
            jnp.where(
                is_abort, 
                loop_par['variables'][1],      # use previous state of the iteration in case of abortion
                _variables_next[1] #
            ),
        )

        if verbose:        
            lax.cond(is_finished, lambda : jax.debug.print("✅ found feasible solution"), lambda : None)

        loop_par = {
            'is_finished' : is_finished,
            'is_abort'    : is_abort,
            'is_X_finite' : is_X_finite,
            'variables'   : variables_next, 
            'parameters'  : loop_par['parameters'], 
            'opt_t'       : opt_t_next, 
            'opt_c_eq'    : opt_c_eq_next, 
            'i'           : loop_par['i'] + 1,
            'verification_state' : verification_state_next, 
            'tol_inner'   : loop_par['tol_inner'], 
            't_final'       : loop_par['t_final'],
        }

        return loop_par
    
    def loop_cond(loop_par):        
        is_n_iter_not_reached = loop_par['i'] < max_iter_boundary_method
        
        is_max_iter_reached_and_not_finished = jnp.logical_and(
            jnp.logical_not(is_n_iter_not_reached),
            jnp.logical_not(loop_par['is_finished']),            
        )
        
        is_continue_iteration = jnp.logical_and(
            jnp.logical_not(loop_par['is_abort']),
            jnp.logical_and(
                jnp.logical_not(loop_par['is_finished']), 
                is_n_iter_not_reached
            )
        )
        
        if verbose:
            lax.cond( loop_par['is_abort'], lambda : jax.debug.print("-> abort as convergence has stopped"), lambda : None)
            if print_errors:
                lax.cond( is_max_iter_reached_and_not_finished,     lambda : jax.debug.print("❌ max. iterations reached without a feasible solution"), lambda : None)
                lax.cond( jnp.logical_not(loop_par['is_X_finite']), lambda : jax.debug.print("❌ found non finite numerics"), lambda : None)
        
        return is_continue_iteration
    
    # loop
    loop_par = {
        'is_finished' : jnp.array(False, dtype=jnp.bool_),
        'is_abort'    : jnp.array(False, dtype=jnp.bool_),
        'is_X_finite' : jnp.array(True, dtype=jnp.bool_),
        'variables'     : variables, 
        'parameters' : parameters, 
        'opt_t' : opt_t, 
        'opt_c_eq' : opt_c_eq, 
        'i' : i, 
        'verification_state' : verification_state_init, 
        'tol_inner' : tol_inner, 
        't_final' : t_final,
    }

    loop_par = lax.while_loop( loop_cond, loop_body, loop_par ) # loop

    n_iter = loop_par['i']

    return loop_par['variables'], loop_par['opt_t'], loop_par['opt_c_eq'], n_iter, loop_par['verification_state']


@partial(jit, static_argnums=(0, 1, 2, 3, 4,  9, 10, 11,  18, 19 ) )
def optimize_trajectory(
    f, 
    g,
    terminal_state_eq_constraints,
    inequ_constraints,
    running_cost,
    
    x0, 
    X_guess,
    U_guess, 
    theta,
    
    max_iter_boundary_method = 40,
    max_iter_inner = 5000,
    verbose = True,
    
    c_eq_init = 100.0,
    opt_t_init = 0.5, 
    lam = 1.6,
    
    eq_tol  = 0.0001,
    t_final = 100.0,
    tol_inner = 0.0001,

    enable_float64 = True,
    max_float32_iterations = 0
):
    """
        Find the optimal control sequence for a given dynamic system, cost function and constraints
        
        Args:
        
            -- callback functions that describe the problem to solve --
            
            f: 
                the discrete-time system function with the prototype x_next = f(x, u, k, theta)
                - x: (n_states, )     the state vector
                - u: (n_inputs, )     the system input(s)
                - k: scalar           the sampling index
                - theta: (JAX-pytree) the parameters theta as passed to optimize_trajectory
            g: 
                the optional output function g(x, u, k, theta)
                - the parameters of the callback have the same meaning like with the callback f
            
            terminal_state_eq_constraints:
                function to evaluate the terminal constraints

            running_cost: 
                function to evaluate the running costs J = running_cost(x, u, t, theta)
                
            inequ_constraints: 
                a function to evaluate the inequality constraints and prototype 
                c_neq = inequ_constraints(x, u, k, theta)
                
                A fulfilled constraint is indicated by a the value c_neq[] >= 0.
                
                
            -- dynamic parameters (jax values) --
                
            x0:
                a vector containing the initial state of the system described by f
            
            X_guess: (n_steps, n_states)
                an initial guess for a solution to the optimal state trajectory
                
            U_guess: (n_steps, n_inputs)
                an initial guess for a solution to the optimal sequence of control variables
            
            theta: (JAX-pytree)
                parameters to the system model that are forwarded to f, g, cost_fn
                        
                        
            -- static parameters (no jax datatypes) --
            
            max_iter_boundary_method: int
                The maximum number of iterations to apply the boundary method.
                
            max_iter_inner: int
                xxx
                
            verbose: bool
                If true print some information on the solution process

            
            -- solver settings (can be jax datatypes) --
            
            c_eq_init: float
                xxx
                
            opt_t_init: float
                xxx
                
            lam: float
                xxx
                        
            eq_tol: float
                tolerance to maximal error of the equality constraints (maximal absolute error)
                
            t_final: float
                XXXX
                
            tol_inner: float
                tolerance passed to the inner solver

            enable_float64: bool
                use 64-bit floating point if true enabling better precision (default = True)

            max_float32_iterations: int
                apply at max max_float32_iterations number of iterations using 32-bit floating
                point precision enabling faster computation (default = 0)
            
            
        Returns: X_opt, U_opt, system_outputs, res
            X_opt: the optimized state trajectory
            U_opt: the optimized control sequence
            
            system_outputs: 
                The return value of the function g evaluated for X_opt, U_opt
            
            res: solver-internal information that can be unpacked with unpack_res()
            
    """
    
    # check for correct parameters
    assert len(X_guess.shape) == 2
    assert len(U_guess.shape) == 2
    assert len(x0.shape) == 1
        
    n_steps = U_guess.shape[0]
    n_states = x0.shape[0]
    n_inputs = U_guess.shape[1]
    
    assert U_guess.shape[0] == n_steps
    assert n_inputs >= 1
    
    assert X_guess.shape[0] == n_steps
    assert X_guess.shape[1] == n_states
    
    assert type(max_iter_boundary_method) is int
    assert type(max_iter_inner) is int
    
    #
    if verbose:
        jax.debug.print("👉 solving problem with n_horizon={n_steps}, n_states={n_states} n_inputs={n_inputs}", 
                        n_steps=n_steps, n_states=n_states, n_inputs=n_inputs)
    
    # index vector
    K = jnp.arange(n_steps)

    # pack parameters and variables
    parameters = (K, theta, x0, )
    static_parameters = (f, terminal_state_eq_constraints, inequ_constraints, running_cost)
    variables = (X_guess, U_guess)

    # pass static parameters into objective function
    objective_ = partial(objective_penality_method, static_parameters=static_parameters)
    feasibility_metric_ = partial(feasibility_metric_penality_method, static_parameters=static_parameters)

    # verification function (non specific to given problem to solve)
    verification_fn_ = partial(
        _verify_step, 
        feasibility_metric_fn=feasibility_metric_, t_final=t_final, eq_tol=eq_tol, verbose=verbose
    )
    
    # trace vars
    trace_init = ( 
        math.nan*jnp.zeros(max_iter_boundary_method, dtype=jnp.float32),
        math.nan*jnp.zeros(max_iter_boundary_method, dtype=jnp.float32), 
        -jnp.ones(max_iter_boundary_method, dtype=jnp.int32), 
    )

    #
    # iterate
    #

    opt_t = opt_t_init
    opt_c_eq = c_eq_init
    i = 0
    verification_state = (trace_init, jnp.array(0, dtype=jnp.bool_) )

    # float32
    if max_float32_iterations > 0:
        variables, opt_t, opt_c_eq, n_iter_f32, verification_state = _optimize_trajectory( 
            i, 
            variables, parameters, 
            jnp.array(opt_t, dtype=jnp.float32),
            jnp.array(opt_c_eq, dtype=jnp.float32),
            verification_state, lam,
            tol_inner,
            t_final, 
            max_float32_iterations,
            max_iter_inner, objective_, verification_fn_,
            verbose, 
            False, # show_errors
            target_dtype=jnp.float32
        )

        i = i + n_iter_f32

        if verbose:
            jax.debug.print("👉 switching to higher numerical precision after {n_iter_f32} iterations: float32 --> float64", n_iter_f32=n_iter_f32)

    # float64
    if enable_float64:
        variables, opt_t, opt_c_eq, n_iter_f64, verification_state = _optimize_trajectory( 
            i, 
            variables, parameters, 
            jnp.array(opt_t, dtype=jnp.float64),
            jnp.array(opt_c_eq, dtype=jnp.float64),
            verification_state, lam,
            tol_inner, 
            t_final, 
            max_iter_boundary_method - i, 
            max_iter_inner, objective_, verification_fn_,
            verbose, 
            True if verbose else False, # show_errors
            target_dtype=jnp.float64
        )
        i = i + n_iter_f64


    n_iter = i
    variables_star = variables
    trace = verification_state[0]

    is_converged = verification_state[1]

    #
    # end iterate
    #

    # unpack results for optimized variables
    X_opt, U_opt = variables_star
    
    # evaluate the constraint functions one last time to return the residuals 
    c_eq   = eq_constraint(f, terminal_state_eq_constraints, X_opt, U_opt, K, x0, theta, 0)
    c_ineq = inequ_constraints(X_opt, U_opt, K, theta)
    
    # compute systems outputs for the optimized trajectory
    system_outputs = None
    if g is not None:
        g_ = jax.vmap(g, in_axes=(0, 0, 0, None))
        system_outputs = g_(X_opt, U_opt, K, theta)
        
    # collect results
    res = {
        'is_converged' : is_converged,
        'n_iter' : n_iter,
        'c_eq' : c_eq,
        'c_ineq' : c_ineq,
        'trace' : trace,
        'trace_metric_c_eq' : trace[0],
        'trace_metric_c_ineq' : trace[1],
    }

    return jnp.vstack(( x0, X_opt )), U_opt, system_outputs, res


def _convert_dtype_to_32(pytree):
    """
    """
    
    def _convert_dtype_to_32(x):
        
        if not isinstance(x, jnp.ndarray):
            return jnp.float32
        
        dtype = x.dtype
        
        if dtype == jnp.float64:
            return jnp.float32
        
        if dtype == jnp.float32:
            return jnp.float32
        
        elif dtype == jnp.int64:
            return jnp.int32
        
        elif dtype == jnp.int32:
            return jnp.int32
    
    return jax.tree_map( 
        lambda x: jnp.array(x, dtype=_convert_dtype_to_32(x)), 
        pytree 
    )

def _convert_dtype_to_64(pytree):
    """
    """
    
    def _convert_dtype_to_64(x):
        
        if not isinstance(x, jnp.ndarray):
            return jnp.float64
        
        dtype = x.dtype
        
        if dtype == jnp.float64:
            return jnp.float64
        
        if dtype == jnp.float32:
            return jnp.float64
        
        elif dtype == jnp.int64:
            return jnp.int64
        
        elif dtype == jnp.int32:
            return jnp.int64
    
    return jax.tree_map( 
        lambda x: jnp.array(x, dtype=_convert_dtype_to_64(x)), 
        pytree 
    )

def _convert_dtype(pytree, target_dtype = jnp.float32 ):
    """
        t = {
            'a': jnp.array([1,-0.5], dtype=jnp.float64),
            'b': jnp.array([1,-0.5], dtype=jnp.int32),
            'c': 0.1
        }

        _convert_dtype(t, jnp.float32), _convert_dtype(t, jnp.float64)
    """
    if target_dtype == jnp.float32:
        return _convert_dtype_to_32(pytree)
    
    elif target_dtype == jnp.float64:
        return _convert_dtype_to_64(pytree)



class Solver:
    def __init__(self, problem_def_fn, use_continuation=False):
        self.problem_def_fn = problem_def_fn
        
        (
            self.f, self.g, self.running_cost, 
            self.terminal_state_eq_constraints, self.inequ_constraints, 
            self.theta, self.x0, self.make_guess
        ) = problem_def_fn()

        self.max_iter_boundary_method = 40
        self.max_iter_inner = 5000
        self.verbose = True
                
        self.c_eq_init = 100
        self.opt_t_init    = 0.5
        self.lam = 1.6

        self.eq_tol  = 0.0001
        self.t_final = 100.0
        
        self.tol_inner = 0.0001
        
        # get n_steps
        X_guess, _   = self.make_guess(self.x0, self.theta)
        self.n_steps = X_guess.shape[0]
        
        # make memory for guess
        self.X_guess, self.U_guess = self.make_guess(self.x0, self.theta)
        
        self.use_continuation = use_continuation

        self.enable_float64 = True
        self.max_float32_iterations = 0

        # status of latest run
        self.success = False
        self.X_opt = None
        self.U_opt = None
        self.system_outputs = None
        
    def run(self):
        
        # run
        if not self.use_continuation:
            self.X_guess, self.U_guess = self.make_guess(self.x0, self.theta)
        
        start_time = time.time()
        solver_return = optimize_trajectory(
            self.f, 
            self.g,
            self.terminal_state_eq_constraints,
            self.inequ_constraints,
            self.running_cost,
            self.x0,
            
            X_guess       = self.X_guess,
            U_guess       = self.U_guess, 
            theta         = self.theta,
            
            max_iter_boundary_method = self.max_iter_boundary_method,
            max_iter_inner           = self.max_iter_inner,
            verbose                  = self.verbose,
            
            c_eq_init = self.c_eq_init,
            opt_t_init    = self.opt_t_init,
            lam           = self.lam,
            eq_tol        = self.eq_tol,
            t_final         = self.t_final,
            tol_inner     = self.tol_inner,

            enable_float64         = self.enable_float64,
            max_float32_iterations = self.max_float32_iterations,
        )
        end_time = time.time()
        elapsed = end_time - start_time

        if self.verbose:
            print(f"time to run: {elapsed} seconds")
        
        X_opt, U_opt, system_outputs, res = solver_return
        
        self.X_guess = X_opt[1:]
        self.U_guess = U_opt

        self.X_opt = X_opt
        self.U_opt = U_opt
        self.system_outputs = system_outputs
        self.success = res['is_converged'].tolist()
        
        return solver_return
    

def unpack_res(res):
    """
        is_converged, c_eq, c_ineq, trace, n_iter = unpack_res(res)
    """
    is_converged = res['is_converged']
    c_eq = res['c_eq'] 
    c_ineq = res['c_ineq']
    trace = res['trace']
    n_iter = res['n_iter']
    
    return is_converged, c_eq, c_ineq, trace, n_iter
    
    
    
    
    
    
    
