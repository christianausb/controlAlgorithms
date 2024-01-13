import jax
from jax import jit
from jax import lax
import jax.numpy as jnp
import jaxopt

from functools import partial
from inspect import signature

from jax_control_algorithms.common import *
from jax_control_algorithms.jax_helper import *
import time

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Functions:
    f: Callable
    initial_guess: Callable
    g: Callable = None
    terminal_constraints: Callable = None
    inequality_constraints: Callable = None
    cost: Callable = None
    running_cost: Callable = None
    transform_parameters: Callable = None


@dataclass()
class ProblemDefinition:
    functions: Functions
    x0: jnp.ndarray
    parameters: any = None

    def run(self, x0=None, parameters=None, verbose: bool = False):
        solver_return = optimize_trajectory(
            self.functions,
            self.x0 if x0 is None else x0,
            self.parameters if parameters is None else parameters,
            get_default_solver_settings(),
            enable_float64=True,
            max_float32_iterations=0,
            max_trace_entries=100,
            verbose=verbose,
        )

        return solver_return


@dataclass
class SolverReturn:
    is_converged: bool
    n_iter: jnp.ndarray
    c_eq: jnp.ndarray
    c_ineq: jnp.ndarray
    trace: tuple


def constraint_geq(x, v):
    """
        define 'greater than' inequality constraint
        
        x >= v
    """
    return x - v


def constraint_leq(x, v):
    """
        define 'less than' inequality constraint

        x <= v
    """
    return v - x


def _boundary_fn(x, t_opt, y_max=10, is_continue_linear=False):
    """
        computes the boundary function of x
    """

    # assert y_max > 0

    # which x yields -1/t_opt * log(x) = y_max
    # exp(log(x)) = exp( -y_max * t_opt )
    # AW: x_thr = exp( -y_max * t_opt )

    x_thr = jnp.exp(-y_max * t_opt)

    # what is d/dx (-1/t_opt) * jnp.log(x) with x=x_thr ?
    # AW: (-1/t_opt) * 1/x_thr

    ddx = (-1 / t_opt) * 1 / x_thr

    # linear continuation for x < x_thr (left side)
    if is_continue_linear:
        _ddx = jnp.clip(ddx, -y_max * 10, 0)
        x_linear_cont = _ddx * (x - x_thr) + y_max
    else:
        x_linear_cont = y_max

    x_boundary_fn = -(1 / t_opt) * jnp.log(x)

    #
    y = jnp.where(x < x_thr, x_linear_cont, x_boundary_fn)

    return y


def _eq_constraint(f, terminal_constraints, X_opt_var, U_opt_var, K, x0, parameters, power):
    """
        algebraic constraints for the system dynamics
    """

    X = jnp.vstack((x0, X_opt_var))

    X_next = eval_X_next(f, X[:-1], U_opt_var, K, parameters)

    # compute c_eq( i ) = x( i+1 ) - x_next( i ) for all i
    c_eq_running = jnp.exp2(power) * X[1:] - jnp.exp2(power) * X_next

    if terminal_constraints is not None:
        # terminal constraints are defined
        x_terminal = X_opt_var[-1]

        number_parameters_to_terminal_fn = len(signature(terminal_constraints).parameters)  # TODO: This can be removed
        if number_parameters_to_terminal_fn == 2:
            # the constraint function implements the power parameter

            c_eq_terminal = jnp.exp2(power) * terminal_constraints(x_terminal, parameters)

        elif number_parameters_to_terminal_fn == 3:

            c_eq_terminal = terminal_constraints(x_terminal, parameters, power)

        # total
        c_eq = jnp.vstack((c_eq_running, c_eq_terminal))
    else:
        # no terminal constraints are considered
        c_eq = c_eq_running

    return c_eq


def _vectorize_running_cost(f_rk):
    """ 
        vectorize the running cost function running_cost(x, u, t, parameters)
    """
    return jax.vmap(f_rk, in_axes=(0, 0, 0, None))


def _evaluate_cost(f, cost, running_cost, X, U, K, parameters):
    """
        evaluate the cost of the given configuration X, U
    """

    assert callable(cost) or callable(running_cost), 'no cost function was given'

    zero = jnp.array(0.0, dtype=jnp.float32)
    cost = cost(X, U, K, parameters) if callable(cost) else zero
    running_cost = jnp.sum(_vectorize_running_cost(running_cost)(X, U, K, parameters) if callable(running_cost) else zero)

    assert cost.shape == (), 'return value of the cost function must be a scalar'

    return cost + running_cost


def __objective_penality_method(variables, parameters_passed_to_solver, static_parameters):

    K, parameters, x0, penality_parameter, opt_c_eq = parameters_passed_to_solver
    f, terminal_constraints, inequality_constraints, cost, running_cost = static_parameters
    X, U = variables

    n_steps = X.shape[0]
    assert U.shape[0] == n_steps

    # scaling factor exponent
    power = 0

    # get equality constraint. The constraints are fulfilled of all elements of c_eq are zero
    c_eq = _eq_constraint(f, terminal_constraints, X, U, K, x0, parameters, power).reshape(-1)
    c_ineq = inequality_constraints(X, U, K, parameters).reshape(-1)

    # equality constraints using penality method
    J_equality_costs = opt_c_eq * jnp.mean((c_eq.reshape(-1))**2)

    # eval cost function of problem definition
    J_cost_function = _evaluate_cost(f, cost, running_cost, X, U, K, parameters)

    # apply boundary costs (boundary function)
    J_boundary_costs = jnp.mean(_boundary_fn(c_ineq, penality_parameter, 11, True))

    return J_equality_costs + J_cost_function + J_boundary_costs, c_eq


def _objective_penality_method(variables, parameters, static_parameters):
    return __objective_penality_method(variables, parameters, static_parameters)[0]


def _feasibility_metric_penality_method(variables, parameters_of_dynamic_model, static_parameters):

    K, parameters, x0 = parameters_of_dynamic_model
    f, terminal_constraints, inequality_constraints, cost, running_cost = static_parameters
    X, U = variables

    # get equality constraint. The constraints are fulfilled of all elements of c_eq are zero
    c_eq = _eq_constraint(f, terminal_constraints, X, U, K, x0, parameters, 0)
    c_ineq = inequality_constraints(X, U, K, parameters)

    #
    metric_c_eq = jnp.max(jnp.abs(c_eq))

    # check for violations of the boundary
    metric_c_ineq = jnp.max(-jnp.where(c_ineq > 0, 0, c_ineq))

    neq_tol = 0.0001
    is_solution_inside_boundaries = metric_c_ineq < neq_tol  # check if the solution is inside (or close) to the boundary

    return metric_c_eq, is_solution_inside_boundaries


def _check_monotonic_convergence(i, trace):
    """
        Check the monotonic convergence of the error for the equality constraints 
    """
    trace_data = get_trace_data(trace)

    # As being in the 2nd iteration, compare to prev. metric and see if it got smaller
    is_metric_check_active = i > 2

    def true_fn(par):
        i, trace = par

        delta_max_eq_error = trace[0][i] - trace[0][i - 1]
        is_abort = delta_max_eq_error >= 0

        return is_abort

    def false_fn(par):
        return False

    is_not_monotonic = lax.cond(is_metric_check_active, true_fn, false_fn, (
        i,
        trace_data,
    ))

    return is_not_monotonic


def _verify_step(
    verification_state, i, res_inner, variables, parameters_of_dynamic_model, penality_parameter, feasibility_metric_fn,
    eq_tol, verbose: bool
):
    """
        verify the feasibility of the current state of the solution. This function is executed 
        for each iteration of the outer optimization loop.
    """

    trace, _, = verification_state

    #
    is_X_finite = jnp.isfinite(variables[0]).all()
    is_abort_because_of_nonfinite = jnp.logical_not(is_X_finite)

    # verify step
    max_eq_error, is_solution_inside_boundaries = feasibility_metric_fn(variables, parameters_of_dynamic_model)
    n_iter_inner = res_inner.state.iter_num

    # verify metrics and check for convergence
    is_eq_converged = max_eq_error < eq_tol

    is_converged = jnp.logical_and(
        is_eq_converged,
        is_solution_inside_boundaries,
    )

    # trace
    X, U = variables
    trace_next, is_trace_appended = append_to_trace(
        trace, (max_eq_error, 1.0 * is_solution_inside_boundaries, n_iter_inner, X, U)
    )
    verification_state_next = (trace_next, is_converged,)

    # check for monotonic convergence of the equality constraints
    is_not_monotonic = jnp.logical_and(
        _check_monotonic_convergence(i, trace_next),
        jnp.logical_not(is_converged),
    )

    i_best = None

    is_abort = jnp.logical_or(is_abort_because_of_nonfinite, is_not_monotonic)

    if verbose:
        jax.debug.print(
            "🔄 it={i} \t (sub iter={n_iter_inner})\tt={penality_parameter} \teq_error/eq_tol={max_eq_error} %\tinside bounds: {is_solution_inside_boundaries}",
            i=i,
            penality_parameter=my_to_int(my_round(penality_parameter, decimals=0)),
            max_eq_error=my_to_int(my_round(100 * max_eq_error / eq_tol, decimals=0)),
            n_iter_inner=n_iter_inner,
            is_solution_inside_boundaries=is_solution_inside_boundaries,
        )

        if False:  # additional info (for debugging purposes)
            jax.debug.print(
                "   is_abort_because_of_nonfinite={is_abort_because_of_nonfinite} is_not_monotonic={is_not_monotonic}) " +
                "is_eq_converged={is_eq_converged}, is_solution_inside_boundaries={is_solution_inside_boundaries}",
                is_abort_because_of_nonfinite=is_abort_because_of_nonfinite,
                is_not_monotonic=is_not_monotonic,
                is_eq_converged=is_eq_converged,
                is_solution_inside_boundaries=is_solution_inside_boundaries,
            )

    # verification_state, is_finished, is_abort, i_best
    return verification_state_next, is_converged, is_eq_converged, is_abort, is_X_finite, i_best


def _run_outer_loop(
    i, variables, parameters_of_dynamic_model, penality_parameter_trace, opt_c_eq, verification_state_init, solver_settings,
    objective_fn, verification_fn, verbose, print_errors, target_dtype
):
    """
        Execute the outer loop of the optimization process: herein in each iteration, the parameters of the
        boundary function and the quality cost weight factor are adjusted so that in the final iteration 
        the equality and inequality constraints are fulfilled.
    """

    # convert dtypes
    (
        variables,
        parameters_of_dynamic_model,
        penality_parameter_trace,
        opt_c_eq,
        verification_state_init,
        lam,
        tol_inner,
    ) = convert_dtype(
        (
            variables,
            parameters_of_dynamic_model,
            penality_parameter_trace,
            opt_c_eq,
            verification_state_init,
            solver_settings['lam'],
            solver_settings['tol_inner'],
        ), target_dtype
    )

    # _solver_settings = convert_dtype(solver_settings, target_dtype)

    #
    # loop:
    #

    def loop_body(loop_par):

        # loop iteration variable i
        i = loop_par['i']

        # get the penality parameter
        penality_parameter = loop_par['penality_parameter_trace'][i]
        is_finished_2 = i >= loop_par['penality_parameter_trace'].shape[0] - 1

        #
        parameters_passed_to_inner_solver = loop_par['parameters_of_dynamic_model'] + (
            penality_parameter,
            loop_par['opt_c_eq'],
        )

        # run inner solver
        gd = jaxopt.BFGS(
            fun=objective_fn, value_and_grad=False, tol=loop_par['tol_inner'], maxiter=solver_settings['max_iter_inner']
        )
        res = gd.run(loop_par['variables'], parameters=parameters_passed_to_inner_solver)
        _variables_next = res.params

        # run callback to verify the solution
        verification_state_next, is_finished_1, is_eq_converged, is_abort, is_X_finite, i_best = verification_fn(
            loop_par['verification_state'], i, res, _variables_next, loop_par['parameters_of_dynamic_model'], penality_parameter
        )

        # c_eq-control
        opt_c_eq_next = jnp.where(
            is_eq_converged,

            # in case of convergence of the error below the threshold there is not need to increase c_eq
            loop_par['opt_c_eq'],

            # increase c_eq
            loop_par['opt_c_eq'] * lam,
        )

        # use previous state of the iteration in case of abortion (when is_abort == True)
        variables_next = (
            jnp.where(
                is_abort,
                loop_par['variables'][0],
                _variables_next[0]  #
            ),
            jnp.where(
                is_abort,
                loop_par['variables'][1],  # use previous state of the iteration in case of abortion
                _variables_next[1]  #
            ),
        )

        # solution found?
        is_finished = jnp.logical_and(is_finished_1, is_finished_2)

        if verbose:
            lax.cond(is_finished, lambda: jax.debug.print("✅ found feasible solution"), lambda: None)

        loop_par = {
            'is_finished': is_finished,
            'is_abort': is_abort,
            'is_X_finite': is_X_finite,
            'variables': variables_next,
            'parameters_of_dynamic_model': loop_par['parameters_of_dynamic_model'],
            'penality_parameter_trace': penality_parameter_trace,
            'opt_c_eq': opt_c_eq_next,
            'i': loop_par['i'] + 1,
            'verification_state': verification_state_next,
            'tol_inner': loop_par['tol_inner'],
        }

        return loop_par

    def loop_cond(loop_par):
        is_n_iter_not_reached = loop_par['i'] < solver_settings['max_iter_boundary_method']

        is_max_iter_reached_and_not_finished = jnp.logical_and(
            jnp.logical_not(is_n_iter_not_reached),
            jnp.logical_not(loop_par['is_finished']),
        )

        is_continue_iteration = jnp.logical_and(
            jnp.logical_not(loop_par['is_abort']),
            jnp.logical_and(jnp.logical_not(loop_par['is_finished']), is_n_iter_not_reached)
        )

        if verbose:
            lax.cond(loop_par['is_abort'], lambda: jax.debug.print("-> abort as convergence has stopped"), lambda: None)
            if print_errors:
                lax.cond(
                    is_max_iter_reached_and_not_finished,
                    lambda: jax.debug.print("❌ max. iterations reached without a feasible solution"), lambda: None
                )
                lax.cond(
                    jnp.logical_not(loop_par['is_X_finite']), lambda: jax.debug.print("❌ found non finite numerics"), lambda: None
                )

        return is_continue_iteration

    # loop
    loop_par = {
        'is_finished': jnp.array(False, dtype=jnp.bool_),
        'is_abort': jnp.array(False, dtype=jnp.bool_),
        'is_X_finite': jnp.array(True, dtype=jnp.bool_),
        'variables': variables,
        'parameters_of_dynamic_model': parameters_of_dynamic_model,
        'penality_parameter_trace': penality_parameter_trace,
        'opt_c_eq': opt_c_eq,
        'i': i,
        'verification_state': verification_state_init,
        'tol_inner': tol_inner,
    }

    loop_par = lax.while_loop(loop_cond, loop_body, loop_par)  # loop

    n_iter = loop_par['i']

    return loop_par['variables'], loop_par['opt_c_eq'], n_iter, loop_par['verification_state']


def _solve(
    variables, parameters_of_dynamic_model, solver_settings, trace_init, objective_, verification_fn_, max_float32_iterations,
    enable_float64, verbose
):
    """
        execute the solution finding process
    """

    opt_c_eq = solver_settings['c_eq_init']
    i = 0
    verification_state = (trace_init, jnp.array(0, dtype=jnp.bool_))

    # iterations that are performed using float32 datatypes
    if max_float32_iterations > 0:
        variables, opt_c_eq, n_iter_f32, verification_state = _run_outer_loop(
            i,
            variables,
            parameters_of_dynamic_model,
            solver_settings['penality_parameter_trace'],
            jnp.array(opt_c_eq, dtype=jnp.float32),
            verification_state,
            solver_settings,
            objective_,
            verification_fn_,
            verbose,
            False,  # show_errors
            target_dtype=jnp.float32
        )

        i = i + n_iter_f32

        if verbose:
            jax.debug.print(
                "👉 switching to higher numerical precision after {n_iter_f32} iterations: float32 --> float64",
                n_iter_f32=n_iter_f32
            )

    # iterations that are performed using float64 datatypes
    if enable_float64:
        variables, opt_c_eq, n_iter_f64, verification_state = _run_outer_loop(
            i,
            variables,
            parameters_of_dynamic_model,
            solver_settings['penality_parameter_trace'],
            jnp.array(opt_c_eq, dtype=jnp.float64),
            verification_state,
            solver_settings,
            objective_,
            verification_fn_,
            verbose,
            True if verbose else False,  # show_errors
            target_dtype=jnp.float64
        )
        i = i + n_iter_f64

    n_iter = i
    variables_star = variables
    trace = get_trace_data(verification_state[0])

    is_converged = verification_state[1]

    return variables_star, is_converged, n_iter, trace


def _get_sizes(X_guess, U_guess, x0):
    n_steps = U_guess.shape[0]
    n_states = x0.shape[0]
    n_inputs = U_guess.shape[1]

    return n_steps, n_states, n_inputs


def _verify_shapes(X_guess, U_guess, x0):
    # check for correct parameters
    assert len(X_guess.shape) == 2
    assert len(U_guess.shape) == 2
    assert len(x0.shape) == 1

    n_steps, n_states, n_inputs = _get_sizes(X_guess, U_guess, x0)

    assert U_guess.shape[0] == n_steps
    assert n_inputs >= 1

    assert X_guess.shape[0] == n_steps
    assert X_guess.shape[1] == n_states

    return


def generate_penality_parameter_trace(t_start, t_final, n_steps):
    """
    t_start: Initial penality parameter t of the penality method
    t_final: maximal penality parameter t to apply
    n_steps: the length of the trace
    """
    lam = (t_final / t_start)**(1 / (n_steps - 1))
    t_trace = t_start * lam**jnp.arange(n_steps)
    return t_trace, lam

def get_default_solver_settings():

    solver_settings = {
        'max_iter_boundary_method': 40,
        'max_iter_inner': 5000,
        'c_eq_init': 100.0,
        'lam': 1.6,
        'eq_tol': 0.0001,
        'penality_parameter_trace' : generate_penality_parameter_trace(t_start=0.5, t_final=100.0, n_steps=13)[0],
        'tol_inner': 0.0001,
    }

    return solver_settings


@partial(jit, static_argnums=(0, 4, 5, 6, 7))
def optimize_trajectory(
    # static
    functions: Functions,  # 0

    # dynamic
    x0,  # 1
    parameters,  # 2
    solver_settings,  # 3

    # static
    enable_float64=True,  # 4
    max_float32_iterations=0,
    max_trace_entries=100,
    verbose=True,
):
    """
        Find the optimal control sequence for a given dynamic system, cost function, and constraints

        The penality method is used to implement inequality constraints. Herein using an inner loop
        a standard solver iteratively solves an unconstrained optimization problem. In an outer loop
        the equality and inequality constraints are implemented. Herein, the penality parameter t 
        is increased for each outer iteration to tighten the boundary constraints.
        
        Args:
        
        functions : Functions
            -- a collection of callback functions that describe the problem to solve --
        
            f: 
                the discrete-time system function with the prototype x_next = f(x, u, k, parameters)
                - x: (n_states, )     the state vector
                - u: (n_inputs, )     the system input(s)
                - k: scalar           the sampling index, starts at 0
                - parameters: (JAX-pytree) the parameters parameters as passed to optimize_trajectory
            g: 
                the optional output function g(x, u, k, parameters)
                - the parameters of the callback have the same meaning as the ones of f

            terminal_constraints:
                function to evaluate the terminal constraints

            cost:
                function to evaluate the cost J = cost(X, U, T, parameters)
                Unlike running_cost, the entire vectors for the state X and actuation U trajectories
                are passed.

            running_cost: 
                function to evaluate the running costs J = running_cost(x, u, t, parameters)
                Unlike cost, associated samples of the state (x) and the actuation trajectory (u) 
                are passed.
                
            inequality_constraints: 
                a function to evaluate the inequality constraints and prototype 
                c_neq = inequality_constraints(X, U, K, parameters)
                
                A fulfilled constraint is indicated by a the value c_neq[] >= 0.

            transform_parameters:
                a function (or None) that is called to transform the problem parameters before
                running the optimization, i.e.,

                parameters_transformed = transform_parameters(parameters)

                The transformed parameters are then used for finding the solution.            
                
            initial_guess:
                A function that computes an initial guess for a solution with the prototype

                guess = initial_guess(x0, parameters)

                Herein, guess is a dict with the guessed solutions for X and U the fields as follows
                
                guess = { 'X_guess' : X_guess, 'U_guess' : U_guess }

            
        -- dynamic parameters (jax values) --
            
        x0:
            a vector containing the initial state of the system described by the function f
        
        parameters: (JAX-pytree)
            parameters to the system model that are passed to f, g, running_cost

        solver_settings : dict 
            
            Parameters for the solver in form of a dictionary.
            Default values: default settings are returned by the function get_default_solver_settings()
                    
            Possible fields are:

            max_iter_boundary_method: int
                The maximum number of iterations to apply the boundary method (outer solver loop)

            max_iter_inner: int
                xxx

            c_eq_init: float
                xxx
                
            lam: float
                factor with which the penality parameter is increased in each iteration
                        
            eq_tol: float
                tolerance to maximal error of the equality constraints (maximal absolute error)
                                
            penality_parameter_trace: ndarray
                A list of penality parameters to be successively applied in the iterations
                of the outer solver loop. This list can be, e.g., generated by the function
                generate_penality_parameter_trace.

                Default value:
                    generate_penality_parameter_trace(t_start=0.5, t_final=100.0, n_steps=13)[0]

            tol_inner: float
                tolerance passed to the inner solver



        -- Other static parameters (these are static values in jax jit-compilation) are --
        enable_float64: bool
            use 64-bit floating point if true enabling better precision (default = True)

        max_float32_iterations: int
            apply at max max_float32_iterations number of iterations using 32-bit floating
            point precision enabling faster computation (default = 0)            

        max_trace_entries
            The number of elements in the tracing memory 
            
        verbose: bool
            If true print some information on the solution process


            
        Returns: X_opt, U_opt, system_outputs, res
            X_opt: the optimized state trajectory
            U_opt: the optimized control sequence
            
            system_outputs: 
                The return value of the function g evaluated for X_opt, U_opt
            
            res: solver-internal information that can be unpacked with unpack_res()
    """

    if verbose:
        print('compiling optimizer...')

    #
    if callable(functions.transform_parameters):
        parameters = functions.transform_parameters(parameters)

    assert callable(functions.f), 'a state transition function f must be provided'
    assert callable(
        functions.initial_guess
    ), 'a function initial_guess must be provided that computes an initial guess for the solution'

    initial_guess = functions.initial_guess(x0, parameters)
    X_guess, U_guess = initial_guess['X_guess'], initial_guess['U_guess']

    # verify types and shapes
    _verify_shapes(X_guess, U_guess, x0)

    #
    n_steps, n_states, n_inputs = _get_sizes(X_guess, U_guess, x0)

    # assert type(max_iter_boundary_method) is int
    assert type(max_trace_entries) is int

    #
    if verbose:
        jax.debug.print(
            "👉 solving problem with n_horizon={n_steps}, n_states={n_states} n_inputs={n_inputs}",
            n_steps=n_steps,
            n_states=n_states,
            n_inputs=n_inputs
        )

    # index vector
    K = jnp.arange(n_steps)

    # pack parameters and variables
    parameters_of_dynamic_model = (
        K,
        parameters,
        x0,
    )
    static_parameters = (
        functions.f, functions.terminal_constraints, functions.inequality_constraints, functions.cost, functions.running_cost
    )
    variables = (X_guess, U_guess)

    # pass static parameters into objective function
    objective_ = partial(_objective_penality_method, static_parameters=static_parameters)
    feasibility_metric_ = partial(_feasibility_metric_penality_method, static_parameters=static_parameters)

    # verification function (non specific to given problem to solve)
    verification_fn_ = partial(
        _verify_step, feasibility_metric_fn=feasibility_metric_, eq_tol=solver_settings['eq_tol'], verbose=verbose
    )

    # trace vars
    trace_init = init_trace_memory(
        max_trace_entries, (jnp.float32, jnp.float32, jnp.int32, jnp.float32, jnp.float32),
        (jnp.nan, jnp.nan, -1, jnp.nan * jnp.zeros_like(X_guess), jnp.nan * jnp.zeros_like(U_guess))
    )

    # run solver
    variables_star, is_converged, n_iter, trace = _solve(
        variables, parameters_of_dynamic_model, solver_settings, trace_init, objective_, verification_fn_, max_float32_iterations,
        enable_float64, verbose
    )

    # unpack results for optimized variables
    X_opt, U_opt = variables_star

    # evaluate the constraint functions one last time to return the residuals
    c_eq = _eq_constraint(functions.f, functions.terminal_constraints, X_opt, U_opt, K, x0, parameters, 0)
    c_ineq = functions.inequality_constraints(X_opt, U_opt, K, parameters)

    # compute systems outputs for the optimized trajectory
    system_outputs = None
    if functions.g is not None:
        g_vectorized = jax.vmap(functions.g, in_axes=(0, 0, 0, None))
        system_outputs = g_vectorized(X_opt, U_opt, K, parameters)

    # collect results
    res = {
        'is_converged': is_converged,
        'n_iter': n_iter,
        'c_eq': c_eq,
        'c_ineq': c_ineq,
        'trace': trace,
        'trace_metric_c_eq': trace[0],
        'trace_metric_c_ineq': trace[1],
    }

    return jnp.vstack((x0, X_opt)), U_opt, system_outputs, res


class Solver:
    """
        High-level interface to the solver
    """

    def __init__(self, problem_def_fn):
        self.problem_def_fn = problem_def_fn

        # get problem definition
        self.problem_definition = problem_def_fn()
        assert type(self.problem_definition) is ProblemDefinition

        self.solver_settings = get_default_solver_settings()

        self.enable_float64 = True
        self.max_float32_iterations = 0
        self.verbose = True

        # status of latest run
        self.success = False
        self.X_opt = None
        self.U_opt = None
        self.system_outputs = None

    def run(self):
        start_time = time.time()

        solver_return = optimize_trajectory(
            self.problem_definition.functions,
            self.problem_definition.x0,
            self.problem_definition.parameters,
            self.solver_settings,
            enable_float64=self.enable_float64,
            max_float32_iterations=self.max_float32_iterations,
            max_trace_entries=100,
            verbose=self.verbose,
        )
        end_time = time.time()
        elapsed = end_time - start_time

        if self.verbose:
            print(f"time to run: {elapsed} seconds")

        X_opt, U_opt, system_outputs, res = solver_return

        self.X_opt = X_opt
        self.U_opt = U_opt
        self.system_outputs = system_outputs
        self.success = res['is_converged'].tolist()

        return solver_return


def unpack_res(res):
    """
        Unpack the results of the solver

        is_converged, c_eq, c_ineq, trace, n_iter = unpack_res(res)
    """
    is_converged = res['is_converged']
    c_eq = res['c_eq']
    c_ineq = res['c_ineq']
    trace = res['trace']
    n_iter = res['n_iter']

    traces = {
        'X_trace': trace[3],
        'U_trace': trace[4],
    }

    return is_converged, c_eq, c_ineq, traces, n_iter
