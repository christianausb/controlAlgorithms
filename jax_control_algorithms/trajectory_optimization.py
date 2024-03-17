import jax
from jax import jit
import jax.numpy as jnp

from functools import partial

from jax_control_algorithms.common import *
from jax_control_algorithms.jax_helper import *
import time

from dataclasses import dataclass
from typing import Callable

from jax_control_algorithms.trajectory_optim.dynamics_constraints import eval_dynamics_equality_constraints
from jax_control_algorithms.trajectory_optim.penality_method import *
from jax_control_algorithms.trajectory_optim.outer_loop_solver import run_outer_loop_solver

"""
    Perform trajectory optimization of a dynamic system by finding the control sequence that 
    minimizes a cost function under inequality constraints. Terminal constraints are optional.

"""

@dataclass(frozen=True)
class Functions:
    f: Callable
    initial_guess: Callable = None
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

    def run(self, x0=None, parameters=None, verbose: bool = False, solver_settings=None):
        solver_return = optimize_trajectory(
            self.functions,
            self.x0 if x0 is None else x0,
            self.parameters if parameters is None else parameters,
            get_default_solver_settings() if solver_settings is None else solver_settings,
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


def _get_sizes(U_guess, x0):
    n_steps = U_guess.shape[0]
    n_states = x0.shape[0]
    n_inputs = U_guess.shape[1]

    return n_steps, n_states, n_inputs


def _verify_shapes(X_guess, U_guess, x0):
    # check for correct parameters
    assert len(X_guess.shape) == 2
    assert len(U_guess.shape) == 2
    assert len(x0.shape) == 1

    n_steps, n_states, n_inputs = _get_sizes(U_guess, x0)

    assert U_guess.shape[0] == n_steps
    assert n_inputs >= 1

    assert X_guess.shape[0] == n_steps
    assert X_guess.shape[1] == n_states

    return


def generate_penalty_parameter_trace(t_start, t_final, n_steps):
    """
    Generate a sequence of penalty factors to be used in the optimization process

    Args:
        t_start: Initial penalty parameter t of the penalty method
        t_final: maximal penalty parameter t to apply
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
        'penalty_parameter_trace': generate_penalty_parameter_trace(t_start=0.5, t_final=100.0, n_steps=13)[0],
        'tol_inner': 0.0001,
    }

    return solver_settings


def _transform_parameters(functions, parameters):

    #
    if callable(functions.transform_parameters):
        parameters = functions.transform_parameters(parameters)

    return parameters


def _build_sampling_index_vector(n_steps):
    K = jnp.arange(n_steps)
    return K


def _compute_system_outputs(g, X, U_opt, K, parameters):

    if X.shape[0] == U_opt.shape[0] + 1:
        # cut the latest state vector from which the output can not be computed as no control command u is available here
        _X = X[:-1]
    elif X.shape[0] == U_opt.shape[0]:
        _X = X
    else:
        raise BaseException('invalid number of sampling instances in X and U')

    system_outputs = None
    if g is not None:
        g_vectorized = jax.vmap(g, in_axes=(0, 0, 0, None))
        system_outputs = g_vectorized(_X, U_opt, K, parameters)

    return system_outputs


def compute_system_outputs(
    functions: Functions,
    parameters,
    X,
    U_opt,
):

    n_steps = U_opt.shape[0]

    parameters = _transform_parameters(functions, parameters)
    K = _build_sampling_index_vector(n_steps)
    return _compute_system_outputs(functions.g, X, U_opt, K, parameters)


@partial(jit, static_argnums=(0, 4, 5, 6, 7))
def optimize_trajectory(
    # static
    functions: Functions,

    # dynamic
    x0,
    parameters,
    solver_settings,

    # static
    enable_float64=True,
    max_float32_iterations=0,
    max_trace_entries=100,
    verbose=True,
):
    """
        Find the optimal control sequence for a given dynamic system, cost function, and constraints

        The penalty method is used to implement inequality constraints. Herein using an inner loop
        a standard solver iteratively solves an unconstrained optimization problem. In an outer loop
        the equality and inequality constraints are implemented. Herein, the penalty parameter t 
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
                factor with which the penalty parameter is increased in each iteration
                        
            eq_tol: float
                tolerance to maximal error of the equality constraints (maximal absolute error)
                                
            penalty_parameter_trace: ndarray
                A list of penalty parameters to be successively applied in the iterations
                of the outer solver loop. This list can be, e.g., generated by the function
                generate_penalty_parameter_trace.

                Default value:
                    generate_penalty_parameter_trace(t_start=0.5, t_final=100.0, n_steps=13)[0]

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

    parameters = _transform_parameters(functions, parameters)

    assert callable(functions.f), 'a state transition function f must be provided'
    assert callable(
        functions.initial_guess
    ), 'a function initial_guess must be provided that computes an initial guess for the solution'

    initial_guess = functions.initial_guess(x0, parameters)
    X_guess, U_guess = initial_guess['X_guess'], initial_guess['U_guess']

    # verify types and shapes
    _verify_shapes(X_guess, U_guess, x0)

    #
    n_steps, n_states, n_inputs = _get_sizes(U_guess, x0)

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

    K = _build_sampling_index_vector(n_steps)

    # pack parameters and variables
    parameters_of_dynamic_model = (K, parameters, x0)
    static_parameters = (
        functions.f, functions.terminal_constraints, functions.inequality_constraints, functions.cost, functions.running_cost
    )
    variables = (X_guess, U_guess)

    # pass static parameters into objective function
    objective_ = partial(eval_objective_of_penalty_method, static_parameters=static_parameters)
    feasibility_metric_ = partial(eval_feasibility_metric_of_penalty_method, static_parameters=static_parameters)

    # verification function (non specific to given problem to solve)
    verification_fn_ = partial(
        verify_convergence_of_iteration,
        feasibility_metric_fn=feasibility_metric_,
        eq_tol=solver_settings['eq_tol'],
        verbose=verbose
    )

    # trace vars
    trace_init = init_trace_memory(
        max_trace_entries, (jnp.float32, jnp.float32, jnp.int32, jnp.float32, jnp.float32),
        (jnp.nan, jnp.nan, -1, jnp.nan * jnp.zeros_like(X_guess), jnp.nan * jnp.zeros_like(U_guess))
    )

    # run solver
    variables_star, is_converged, n_iter, trace = run_outer_loop_solver(
        variables, parameters_of_dynamic_model, solver_settings, trace_init, objective_, verification_fn_, max_float32_iterations,
        enable_float64, verbose
    )

    # unpack results for optimized variables
    X_opt, U_opt = variables_star

    # evaluate the constraint functions one last time to return the residuals
    c_eq = eval_dynamics_equality_constraints(functions.f, functions.terminal_constraints, X_opt, U_opt, K, x0, parameters)
    c_ineq = functions.inequality_constraints(X_opt, U_opt, K, parameters)

    X = jnp.vstack((x0, X_opt))

    # compute systems outputs for the optimized trajectory
    system_outputs = _compute_system_outputs(functions.g, X, U_opt, K, parameters)

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

    return X, U_opt, system_outputs, res


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


# https://www.geeksforgeeks.org/typing-namedtuple-improved-namedtuples/
# https://github.com/google/jaxopt/blob/main/jaxopt/_src/base.py
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
