import jax
import jax.numpy as jnp
from jax import lax


def convert_dtype_to_32(pytree):

    def _convert_dtype_to_32(x):

        if not isinstance(x, jnp.ndarray):
            return jnp.float32

        dtype = x.dtype

        if dtype == jnp.float64:
            return jnp.float32

        if dtype == jnp.float32:
            return jnp.float32

        elif dtype == jnp.int64:
            return jnp.int32

        elif dtype == jnp.int32:
            return jnp.int32

    return jax.tree_map(lambda x: jnp.array(x, dtype=_convert_dtype_to_32(x)), pytree)


def convert_dtype_to_64(pytree):

    def _convert_dtype_to_64(x):

        if not isinstance(x, jnp.ndarray):
            return jnp.float64

        dtype = x.dtype

        if dtype == jnp.float64:
            return jnp.float64

        if dtype == jnp.float32:
            return jnp.float64

        elif dtype == jnp.int64:
            return jnp.int64

        elif dtype == jnp.int32:
            return jnp.int64

    return jax.tree_map(lambda x: jnp.array(x, dtype=_convert_dtype_to_64(x)), pytree)


def convert_dtype(pytree, target_dtype=jnp.float32):
    """
        t = {
            'a': jnp.array([1,-0.5], dtype=jnp.float64),
            'b': jnp.array([1,-0.5], dtype=jnp.int32),
            'c': 0.1
        }

        _convert_dtype(t, jnp.float32), _convert_dtype(t, jnp.float64)
    """
    if target_dtype == jnp.float32:
        return convert_dtype_to_32(pytree)

    elif target_dtype == jnp.float64:
        return convert_dtype_to_64(pytree)


def print_if_nonfinite(text: str, x):
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        jax.debug.print(text, x=x)

    lax.cond(is_finite, true_fn, false_fn, x)


def print_if_outofbounds(text: str, x, x_min, x_max, var_to_also_print=None):
    is_oob = jnp.logical_and(
        jnp.all(x > x_min),
        jnp.all(x < x_max),
    )

    def true_fn(x):
        pass

    def false_fn(x):
        # jax.debug.breakpoint()
        jax.debug.print(text, x=x)
        if var_to_also_print is not None:
            jax.debug.print('var_to_also_print={x}', x=var_to_also_print)

    lax.cond(is_oob, true_fn, false_fn, x)


def my_round(x, decimals=2):
    scale = jnp.array(10.0**decimals, dtype=jnp.float64)
    return (jnp.array(jnp.round(scale * x, decimals=0), dtype=jnp.int32) / scale)


def my_to_int(x):
    return jnp.array(x, dtype=jnp.int32)


#
# traces
#


def init_trace_memory(max_trace_entries, dtypes=[jnp.float32, jnp.int32], init_values=[jnp.nan, -1]):

    trace_data = [jnp.repeat(jnp.array([v], dtype=dtype), max_trace_entries, axis=0) for dtype, v in zip(dtypes, init_values)]

    counter = jnp.array(0, dtype=jnp.int32)

    return (counter, trace_data)


def append_to_trace(traces, values_to_append=(1.2, 4)):
    counter = traces[0]
    trace_data = traces[1]

    is_memory_not_full = counter < trace_data[0].shape[0]

    traces_next = jax.tree_map(
        lambda x, y: jnp.where(is_memory_not_full, x, y), _append_to_trace(traces, values_to_append), (counter, trace_data)
    )

    return traces_next, is_memory_not_full


def _append_to_trace(traces, values_to_append):

    counter = traces[0]
    trace_data = traces[1]

    trace_data_next = [trace_data[i].at[counter].set(v) for i, v in enumerate(values_to_append)]

    counter_next = counter + 1

    return (counter_next, trace_data_next)


def get_trace_data(traces):
    trace_data = traces[1]
    return trace_data
