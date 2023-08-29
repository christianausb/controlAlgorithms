import jax
from jax import jit
from jax import lax
from jax import vmap
import jax.numpy as jnp
import jaxopt

from functools import partial
import math
from jax_control_algorithms.common import *

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
        x_linear_cont = ddx * (x - x_thr) + y_max
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
    
def print_if_outofbounds(text : str, x, x_min, x_max):
    is_oob = jnp.logical_and(
        jnp.all( x > x_min ),
        jnp.all( x < x_max ),        
    )
    
    def true_fn(x):
        pass
    def false_fn(x):
        # jax.debug.breakpoint()
        jax.debug.print(text, x=x)

    lax.cond(is_oob, true_fn, false_fn, x)
    
    

#
# routine for state estimation and parameter identification
#

def eq_constraint(f, terminal_state_eq_constraints, X_opt_var, U_opt_var, K, x0, theta):
    """
    algebraic constraints for the system dynamics
    """

    X = jnp.vstack(( x0 , X_opt_var ))
    
    X_next = eval_X_next(f, X[:-1], U_opt_var, K, theta)

    # compute c_eq( i ) = x( i+1 ) - x_next( i ) for all i
    c_eq_running = X[1:] - X_next

    if terminal_state_eq_constraints is not None:
        # terminal constraints are defined
        x_terminal = X_opt_var[-1]
        c_eq_terminal = terminal_state_eq_constraints(x_terminal, theta)
                        
        # total
        c_eq = jnp.vstack( (c_eq_running, c_eq_terminal) )
    else:
        # no terminal contraints are considered
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
    
    K, theta, c_eq_penality, x0, opt_t                                = parameters
    f, terminal_state_eq_constraints, inequ_constraints, running_cost = static_parameters
    X, U                                                              = variables
    
    n_steps = X.shape[0]
    assert U.shape[0] == n_steps

    # get equality constraint. The constraints are fulfilled of all elements of c_eq are zero
    c_eq = eq_constraint(f, terminal_state_eq_constraints, X, U, K, x0, theta).reshape(-1)
    c_ineq = inequ_constraints(X, U, K, theta).reshape(-1)        

    # equality constraints using penality method
    #c_eq_penality = 100 * opt_t
    J1 = c_eq_penality * opt_t * jnp.mean(
        ( c_eq.reshape(-1) )**2
    )
    
    J2 = cost_fn(f, running_cost, X, U, K, theta)
    
    J3 = jnp.mean(
        boundary_fn(c_ineq, opt_t, 11, False)
    )
    
    return J1 + J2 + J3, c_eq


def __feasibility_metric_penality_method(variables, parameters, static_parameters ):
    
    I, theta, c_eq_penality, x0, opt_t                                = parameters
    f, terminal_state_eq_constraints, inequ_constraints, running_cost = static_parameters
    X, U                                                              = variables
    
    # get equality constraint. The constraints are fulfilled of all elements of c_eq are zero
    c_eq = eq_constraint(f, terminal_state_eq_constraints, X, U, I, x0, theta)
    c_ineq = inequ_constraints(X, U, I, theta)
    
    #
    metric_c_eq   = jnp.max(  jnp.abs(c_eq) )
    metric_c_ineq = jnp.max( -jnp.where( c_ineq > 0, 0, c_ineq ) )
    
    return metric_c_eq, metric_c_ineq

def objective_penality_method( variables, parameters, static_parameters ):
    return __objective_penality_method( variables, parameters, static_parameters )[0]

def feasibility_metric_penality_method(variables, parameters, static_parameters ):
    return __feasibility_metric_penality_method(variables, parameters, static_parameters )


@partial(jit, static_argnums=(0, 1, 2, 3, 4, ) )
def plan_trajectory(
    f, 
    g,
    terminal_state_eq_constraints,
    inequ_constraints,
    running_cost,
    
    x0, 
    X_guess,
    U_guess, 
    theta,
    
    c_eq_penality = 100.0,
    opt_t_init = 0.5,
    lam = 1.6,
    
    eq_tol  = 0.0001,
    neq_tol = 0.0001,
    max_iter_boundary_method = 40,
    verbose = True,
):
    """
        Find the optimal control sequence for a given dynamic system, cost function and constraints
        
        Args:
            f: 
                the discrete-time system function with the prototype x_next = f(x, u, k, theta)
                - x: (n_states, )     the state vector
                - u: (n_inputs, )     the system input(s)
                - k: scalar           the sampling index
                - theta: (JAX-pytree) the parameters theta as passed to plan_trajectory
            g: 
                the optional output function g(x, u, k, theta)
                - the parameters of the callback have the same meaning like with the callback f
            
            terminal_state_eq_constraints:
                function to evaluate the terminal constraints

            running_cost: 
                funtion to evaluate the running costs J = running_cost(x, u, t, theta)
                
            inequ_constraints: 
                a function to evaluate the inequality constraints and prototype 
                c_neq = inequ_constraints(x, u, k, theta)
                
                A fulfilled constraint is indicated by a the value c_neq[] >= 0.
                
            x0:
                a vector containing the initial state of the system described by f
            
            X_guess: (n_steps, n_states)
                an initial guess for a solution to the optimal state trajectory
                
            U_guess: (n_steps, n_inputs)
                an initial guess for a solution to the optimal sequence of control variables
            
            theta: (JAX-pytree)
                parameters to the system model that are forwarded to f, g, cost_fn
                        
            eq_tol: float
                tolerance to maximal error of the equality constraints
                
            neq_tol: float
                tolerance to maximal error of the inequality constraints
            
            max_iter_boundary_method: int
                The maximum number of iterations to apply the boundary method.
            
            
        
        Returns: X_opt, U_opt, system_outputs, res
            X_opt: the optimized state trajectory
            U_opt: the optimized control sequence
            
            system_outputs: 
                The return value of the function g evaluated for X_opt, U_opt
            
            res: solver-internal information
            
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
    
    #
    jax.debug.print("👉 solving problem with n_horizon={n_steps}, n_states={n_states} n_inputs={n_inputs}", 
                    n_steps=n_steps, n_states=n_states, n_inputs=n_inputs)
    
    # index vector
    K = jnp.arange(n_steps)

    # pack parameters and variables
    parameters = (K, theta, c_eq_penality, x0, )
    static_parameters = (f, terminal_state_eq_constraints, inequ_constraints, running_cost)
    variables = (X_guess, U_guess)

    # pass static parameters into objective function
    objective_ = partial(objective_penality_method, static_parameters=static_parameters)
    feasibility_metric_ = partial(feasibility_metric_penality_method, static_parameters=static_parameters)
    
    #
    # loop opt_t_init -> opt_t, opt_t = opt_t * 0.xx
    #

    opt_t = opt_t_init
    
    # trace vars
    trace_init = ( 
        math.nan*jnp.zeros(max_iter_boundary_method), 
        math.nan*jnp.zeros(max_iter_boundary_method), 
        -jnp.ones(max_iter_boundary_method, dtype=jnp.int32), 
    )
    
    def loop_body(X):
        _, variables, parameters, opt_t, i, trace = X
            
        #
        parameters_ = parameters + ( opt_t, )

        # run optimization
        gd = jaxopt.BFGS(fun=objective_, value_and_grad=False, tol=0.0001, maxiter=5000)
        res = gd.run(variables, parameters=parameters_)
        variables_star = res.params
        n_iter_inner = res.state.iter_num
        
        if False:
            jax.debug.print(
                "n_iter_inner={n_iter_inner} variables_star={variables_star}", 
                n_iter_inner=n_iter_inner,
                variables_star=variables_star,
            )
        
        # verify step
        metric_c_eq, metric_c_ineq = feasibility_metric_(variables_star, parameters_)
        
        # verify metrics and check for convergence
        is_finished = jnp.logical_and(
            metric_c_eq   < eq_tol,
            metric_c_ineq < neq_tol
        )
        
        # trace
        trace_next = ( 
            trace[0].at[i].set(metric_c_eq),
            trace[1].at[i].set(metric_c_ineq),
            trace[2].at[i].set(n_iter_inner),
        )
        
        if verbose:
            jax.debug.print("🔄 it={i} \t (sub iter={n_iter_inner}) \t t_opt={opt_t} \t  eq={metric_c_eq} \t neq={metric_c_ineq}", 
                            i=i,    opt_t = jnp.round(opt_t, decimals=2),  
                            metric_c_eq   = jnp.round(metric_c_eq, decimals=5), 
                            metric_c_ineq = jnp.round(metric_c_ineq, decimals=5),
                            n_iter_inner  = n_iter_inner ) # ↪
        
            lax.cond(is_finished, lambda : jax.debug.print("✅ found feasible solution"), lambda : None)
        
        return ( is_finished, variables_star, parameters, opt_t * lam, i+1, trace_next )
    
    def loop_cond(X):
        
        is_finished, variables_star, _, _, i, trace = X
        
        is_X_finite = jnp.isfinite(variables_star[0]).all()
        is_n_iter_not_reaced = i < max_iter_boundary_method
        
        is_max_iter_reached_and_not_finished = jnp.logical_and(
            jnp.logical_not(is_n_iter_not_reaced),
            jnp.logical_not(is_finished),            
        )
        
        is_continue = jnp.logical_and(
            is_X_finite,
            jnp.logical_and(
                jnp.logical_not(is_finished), 
                is_n_iter_not_reaced
            )
        )
        
        if verbose:
            lax.cond( is_max_iter_reached_and_not_finished, lambda : jax.debug.print("❌ max. iterations reached without a feasible solution"), lambda : None)
            lax.cond( jnp.logical_not(is_X_finite),         lambda : jax.debug.print("❌ abort because of non finite numerics"), lambda : None)
        
        return is_continue
        
    
    # loop
    X = ( jnp.array(False, dtype=jnp.bool_), variables, parameters, opt_t, 0, trace_init ) # pack
    X = lax.while_loop( loop_cond, loop_body, X ) # loop
    _, variables_star, parameters, opt_t, n_iter, trace = X # unpack
    
    # unpack results for optimized variables
    X_opt, U_opt = variables_star
    
    # evaluate the constraint functions one last time to return the residuals 
    c_eq   = eq_constraint(f, terminal_state_eq_constraints, X_opt, U_opt, K, x0, theta)
    c_ineq = inequ_constraints(X_opt, U_opt, K, theta)
    
    # compute systems outputs for the optimized trajectory
    system_outputs = None
    if g is not None:
        g_ = jax.vmap(g, in_axes=(0, 0, 0, None ) )
        system_outputs = g_(X_opt, U_opt, K, theta)

    # eval metrics and look for convergence
    metric_c_eq_final   = trace[0][n_iter-1]
    metric_c_ineq_final = trace[1][n_iter-1]
    
    is_converged = jnp.logical_and(
        metric_c_eq_final   < eq_tol,
        metric_c_ineq_final < neq_tol
    )
        
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


class Solver:
    def __init__(self, problem_def_fn):
        self.problem_def_fn = problem_def_fn
        
        (
            self.f, self.g, self.running_cost, 
            self.terminal_state_eq_constraints, self.inequ_constraints, 
            self.theta, self.x0, self.make_guess
        ) = problem_def_fn()
                
        self.c_eq_penality = 100
        self.opt_t_init    = 0.5 
        
        # get n_steps
        X_guess, _   = self.make_guess(self.x0, self.theta)
        self.n_steps = X_guess.shape[0]
        
    def run(self):
        
        X_guess, U_guess = self.make_guess(self.x0, self.theta)
        # run
        
        return plan_trajectory(
            self.f, 
            self.g,
            self.terminal_state_eq_constraints,
            self.inequ_constraints,
            self.running_cost,
            self.x0,
            X_guess       = X_guess,
            U_guess       = U_guess, 
            theta         = self.theta,
            c_eq_penality = self.c_eq_penality,
            opt_t_init    = self.opt_t_init,
        )

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
    
    
    
    
    
    
    
