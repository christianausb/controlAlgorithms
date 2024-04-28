from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple
from jax import numpy as jnp

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
    trace : any
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

