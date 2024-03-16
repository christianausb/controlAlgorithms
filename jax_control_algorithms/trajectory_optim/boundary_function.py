import jax.numpy as jnp


def boundary_fn(x, t, y_max=10, is_continue_linear=False):
    """
        computes the boundary function of x that implements a soft-constraint 
        
        Computes the boundary function y(x) = -(1 / t) * log(x) as part of the penality-method
        for implementing constrained optimization.

        Args:
            x: a vector of elements for which to compute the boundary function
            t_opt: describes the 'sharpness' of the soft boundary
            y_max: the value at which y is clipped.
            is_continue_linear: 

    """

    # assert y_max > 0

    # which x yields -1/t_opt * log(x) = y_max
    # exp(log(x)) = exp( -y_max * t_opt )
    # AW: x_thr = exp( -y_max * t_opt )

    x_thr = jnp.exp(-y_max * t)

    # what is d/dx (-1/t_opt) * jnp.log(x) with x=x_thr ?
    # AW: (-1/t_opt) * 1/x_thr

    ddx = (-1 / t) * 1 / x_thr

    # linear continuation for x < x_thr (left side)
    if is_continue_linear:
        _ddx = jnp.clip(ddx, -y_max * 10, 0)
        x_continuation = _ddx * (x - x_thr) + y_max
    else:
        x_continuation = y_max

    x_boundary_fn = -(1 / t) * jnp.log(x)

    #
    y = jnp.where(x < x_thr, x_continuation, x_boundary_fn)

    return y
