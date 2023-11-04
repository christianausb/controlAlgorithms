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
    
    return jax.tree_map( 
        lambda x: jnp.array(x, dtype=_convert_dtype_to_32(x)), 
        pytree 
    )

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
    
    return jax.tree_map( 
        lambda x: jnp.array(x, dtype=_convert_dtype_to_64(x)), 
        pytree 
    )

def convert_dtype(pytree, target_dtype = jnp.float32 ):
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


def print_if_nonfinite(text : str, x):
    is_finite = jnp.isfinite(x).all()
    
    def true_fn(x):
        pass
    def false_fn(x):
        jax.debug.print(text, x=x)

    lax.cond(is_finite, true_fn, false_fn, x)
    
def print_if_outofbounds(text : str, x, x_min, x_max, var_to_also_print=None):
    is_oob = jnp.logical_and(
        jnp.all( x > x_min ),
        jnp.all( x < x_max ),        
    )
    
    def true_fn(x):
        pass
    def false_fn(x):
        # jax.debug.breakpoint()
        jax.debug.print(text, x=x)
        if var_to_also_print is not None:
            jax.debug.print('var_to_also_print={x}', x=var_to_also_print)

    lax.cond(is_oob, true_fn, false_fn, x)
    