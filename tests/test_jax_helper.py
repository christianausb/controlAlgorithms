import jax.numpy as jnp
from jax_control_algorithms.jax_helper import *


def test_tracing_scalar_values():

   # set-up
   traces = init_trace_memory(10, dtypes = [jnp.float32, jnp.int32], init_values=[-10, -1])
   trace_data = get_trace_data(traces)

   # assert
   assert jnp.all( trace_data[0] == -10 )
   assert jnp.all( trace_data[1] == -1 )

   # act
   traces, is_ok = append_to_trace(traces, (1.1, 2))
   trace_data = get_trace_data(traces)

   # assert
   assert is_ok
   assert trace_data[0][0] == 1.1
   assert trace_data[1][0] == 2

   for i in range(9):
      traces, is_ok = append_to_trace(traces, (1.1, 2))
      assert is_ok

   traces, is_ok = append_to_trace(traces, (1.1, 2))
   assert not is_ok

def test_tracing_of_arrays():

   # set-up
   test_array = jnp.arange(0,5)
   traces = init_trace_memory(10, dtypes = [jnp.float32, ], init_values=[ test_array, ])
   trace_data = get_trace_data(traces)

   assert all([
      jnp.all((trace_data[0][i] - test_array) == 0).item()
      for i in range(10)
   ])

   traces, is_ok = append_to_trace(traces, ( jnp.arange(10,15), ))
   trace_data = get_trace_data(traces)
                                 
   assert jnp.all( trace_data[0][0] == jnp.arange(10,15) )
