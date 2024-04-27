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
