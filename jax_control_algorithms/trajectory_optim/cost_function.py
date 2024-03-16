import jax
import jax.numpy as jnp


def _vectorize_running_cost(f_rk):
    """ 
        vectorize the running cost function running_cost(x, u, t, parameters)
    """
    return jax.vmap(f_rk, in_axes=(0, 0, 0, None))


def evaluate_cost(f, cost, running_cost, X, U, K, parameters):
    """
        evaluate the cost of the given configuration X, U
    """

    assert callable(cost) or callable(running_cost), 'no cost function was given'

    zero = jnp.array(0.0, dtype=jnp.float32)
    cost = cost(X, U, K, parameters) if callable(cost) else zero
    running_cost = jnp.sum(_vectorize_running_cost(running_cost)(X, U, K, parameters) if callable(running_cost) else zero)

    assert cost.shape == (), 'return value of the cost function must be a scalar'

    return cost + running_cost
