import jax.numpy as jnp
from jax_control_algorithms.common import eval_X_next
from typing import Callable


def eval_dynamics_equality_constraints(f: Callable, terminal_constraints: Callable, X, U, K, x0, parameters):
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
    h_dynamics_constraints = X[1:] - X_next

    if terminal_constraints is not None:
        # terminal constraints are defined
        x_terminal = X_next[-1]
        h_terminal_constraints = terminal_constraints(x_terminal, parameters)

        # total
        return jnp.vstack((h_dynamics_constraints, h_terminal_constraints))

    # no terminal constraints are considered
    return h_dynamics_constraints

