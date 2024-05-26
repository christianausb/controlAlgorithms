from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple
from jax import numpy as jnp

#from jax_control_algorithms.trajectory_optim.penality_method import generate_penalty_parameter_trace

def _generate_penalty_parameter_trace(t_start, t_final, n_steps):
    lam = (t_final / t_start)**(1 / (n_steps - 1))
    t_trace = t_start * lam**jnp.arange(n_steps)
    return t_trace, lam

def generate_penalty_parameter_trace(t_start=0.5, t_final=100, n_steps=13):
    """
    Generate a sequence of penalty factors to be used in the optimization process

    Args:
        t_start: Initial penalty parameter t of the penalty method
        t_final: maximal penalty parameter t to apply
        n_steps: the length of the trace
    """
    return _generate_penalty_parameter_trace(t_start, t_final, n_steps)[0]


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


class ParametersOfModelToSolve(NamedTuple):
    K: jnp.ndarray = None
    parameters: jnp.ndarray = None
    x0: jnp.ndarray = None


#@dataclass(frozen=True)
class ModelToSolve(NamedTuple):
    functions: Functions = None
    parameters_of_dynamic_model: ParametersOfModelToSolve = None

    #K: jnp.ndarray = None
    #parameters: jnp.ndarray = None
    #x0: jnp.ndarray = None


class ConvergenceControllerState(NamedTuple):
    trace: any
    is_converged: jnp.ndarray


class OuterLoopVariables(NamedTuple):
    is_finished: jnp.ndarray
    is_abort: jnp.ndarray
    is_X_finite: jnp.ndarray
    variables: any
    parameters_of_dynamic_model: any
    penalty_parameter_trace: jnp.ndarray
    opt_c_eq: jnp.ndarray
    lam: jnp.ndarray
    i: jnp.ndarray
    controller_state: ConvergenceControllerState
    tol_inner: jnp.ndarray


class SolverSettings(NamedTuple):
    max_iter_boundary_method : int = 40
    max_iter_inner:float = 5000
    c_eq_init :float = 100.0
    eq_tol :float = 0.0001
    penalty_parameter_trace : jnp.array = generate_penalty_parameter_trace(t_start=0.5, t_final=100.0, n_steps=13)
    tol_inner:float = 0.0001
