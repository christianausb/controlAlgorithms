import jax.numpy as jnp
from inspect import signature
from jax_control_algorithms.common import eval_X_next
from typing import Callable


def eval_dynamics_equality_constraints(f: Callable, terminal_constraints: Callable, X, U, K, x0, parameters, power):
    """
        evaluate the algebraic constraints that form the dynamics described by the transition function

        evaluate 
            c_eq(i) = x(i+1) - x_next(i) for all i, where
            x(i+1) is the one-step ahead prediction of x(i) using f and u(i)

        In case X is a trajectory of the system f, all elements of c_eq are zero.

        Further, terminal_constraints(x_terminal, parameters) is evaluated, wherein
        x_terminal = X[-1] (the last element in the sequence X)
        
        Args:
            f: The discrete-time transition function
            terminal_constraints: A function to eval optional terminal constraints
            X: the samples of the state trajectory candidate
            U: the samples for the control input
            K: sampling indices vector to be passed to f
            x0: The initial state of the trajectory
            parameters: parameters passed to f            
    """

    X = jnp.vstack((x0, X))

    X_next = eval_X_next(f, X[:-1], U, K, parameters)

    # compute c_eq(i) = x(i+1) - x_next(i) for all i
    c_eq_running = jnp.exp2(power) * X[1:] - jnp.exp2(power) * X_next

    if terminal_constraints is not None:
        # terminal constraints are defined
        x_terminal = X[-1]

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
