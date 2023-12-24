import tensorflow as tf
import numpy as np
import jax
from jax import jit
from jax import lax
import jax.numpy as jnp
from jax.experimental import jax2tf
jax.config.update('jax_enable_x64', True)


from jax_control_algorithms.estimation import *

#
# Pendulum model without friction
#
def pendulum_dynamics(x, u, t, theta):
    del t
    assert x.shape[0] == 2
    assert theta.shape[0] == 3

    phi, phi_dot = x    
    a, b, y_ofs = theta

    dxdt = jnp.array([
        phi_dot,
        - a * jnp.sin(phi) - b * phi_dot
    ])
    
    return dxdt

def pendulum_output(x, u, t, theta):
    del t
    phi, phi_dot = x
    a, b, y_ofs = theta
    
    return jnp.array( [phi + y_ofs] )


def pendulum_output2(x, u, t, theta):
    del t
    phi, phi_dot = x
    a, b, y_ofs = theta
    
    return jnp.array([
        phi + y_ofs,    # angle
        phi_dot,        # angle dot
    ])




#
# Pendulum simulation
#

def simulate_pendulum( parameters, initial_states, max_time : float = 3.0, dt : float = 0.01 ):

    n_steps = get_n_steps(max_time, dt)
    U_excitation = jnp.zeros( ( n_steps, 0) ) # no control input
    
    T, X_gt, Y = simulate_dscr(
        f = rk4(pendulum_dynamics, dt), 
        g = pendulum_output, 
        x0 = initial_states,
        U = U_excitation, 
        dt = dt, 
        theta = parameters,
    )
    
    return T, X_gt, Y, U_excitation

def pendulum_estimation_jax(Y_measurement, max_time : float = 3.0, dt : float  = 0.01):
    
    n_steps = get_n_steps(max_time, dt)
    U_excitation = jnp.zeros( ( n_steps, 1) )
    
    wx = jnp.array([ 10.0, 10.0 ]) # high trust into the correctness of the model equations
    wy = jnp.array([ 0.1,       ]) # less trust into the correct measurement of the output as it is subjected to noise

    Wx, Wy = make_const_weights(wx, wy, n_steps)

    X_hat, theta_hat, Y_hat, J_star, res = estimate(
        
        f = rk4(pendulum_dynamics, dt),
        g = pendulum_output, 
        T = jnp.arange( n_steps ) * dt,
        U = U_excitation,
        Y = Y_measurement, 
        Wx = Wx, 
        Wy = Wy, 
        X0=jnp.zeros(( n_steps, 2 )), 
        theta0=jnp.array([4.01, 0.0, 0.0])
    )    
    
    return X_hat.astype(jnp.float32), theta_hat.astype(jnp.float32), Y_hat.astype(jnp.float32), J_star.astype(jnp.float32)

def build_pendulum_estimation_tf_function(max_time : float, dt : float ):
    n_steps = get_n_steps(max_time, dt)

    tf_fn = tf.function(
        jax2tf.convert(
            partial(pendulum_estimation_jax, max_time=max_time, dt=dt)
        ), autograph=False,
        input_signature=[
            tf.TensorSpec(shape=(n_steps, 1), dtype=tf.float32, name='Y_measurement'),
        ],    
    )
    
    return tf_fn

#
# Combined parameter and state estimation
#


def pendulum_model_estimate_objective(Y_measurement, X, theta, lam, wy, wx1, wx2, max_time, dt):
    """
        Evaluates the pendulum model-fit loss function for a given measured output and a 
        state-trajectory / parameter combination.

        Y_measurement is the time-series of the pendulum angle if with_output_psi_dot = True

    """


    n_steps = get_n_steps(max_time, dt)
    U_excitation = jnp.zeros( ( n_steps, 1) )

    lam_Wy  = lam
    lam_Wx1 = lam
    lam_Wx2 = lam
    
    Wy = wy * make_exp_decay_weights(lam_Wy, n_steps)
    Wx = jnp.stack((
        wx1 * make_exp_decay_weights(lam_Wx1, n_steps-1)[:,0],
        wx2 * make_exp_decay_weights(lam_Wx2, n_steps-1)[:,0],
    )).T  
    
    J, X_next, Y_hat = estimate_objective(
        f      = rk4(pendulum_dynamics, dt),
        g      = pendulum_output,

        T      = jnp.arange( n_steps ) * dt,
        U      = U_excitation,
        Y      = Y_measurement, 
        Wx     = Wx,
        Wy     = Wy, 
        X      = X,
        theta  = theta,
    )
    
    return J.astype(jnp.float32), X_next.astype(jnp.float32), Y_hat.astype(jnp.float32)

def build_pendulum_model_estimate_objective_tf_function(wy, wx1, wx2, max_time, dt):
    n_steps = get_n_steps(max_time, dt)
    
    tf_fn = tf.function(
        jax2tf.convert(
            partial(pendulum_model_estimate_objective, wy=wy, wx1=wx1, wx2=wx2, max_time=max_time, dt=dt)
        ), autograph=False,
        input_signature=[
            tf.TensorSpec(shape=(n_steps, 1), dtype=tf.float32, name='Y_measurement'),
            tf.TensorSpec(shape=(n_steps, 2), dtype=tf.float32, name='X'),
            tf.TensorSpec(shape=(3, ),        dtype=tf.float32, name='parameters'),
            tf.TensorSpec(shape=(   ),        dtype=tf.float32, name='lam'),
        ],
    )
    
    return tf_fn

def pendulum_model_estimate_objective2(Y_measurement, X, theta, lam, wy1, wy2, wx1, wx2, max_time, dt):
    """
        Evaluates the pendulum model-fit loss function for a given measured output and a 
        state-trajectory / parameter combination.

        Y_measurement is the time-series of the pendulum angle and its derivative

    """

    assert Y_measurement.shape[1] == 2

    n_steps = get_n_steps(max_time, dt)
    U_excitation = jnp.zeros( ( n_steps, 1) )

    lam_Wy  = lam
    lam_Wx1 = lam
    lam_Wx2 = lam
    
    Wy = jnp.stack((
        wy1 * make_exp_decay_weights(lam_Wy, n_steps)[:,0],
        wy2 * make_exp_decay_weights(lam_Wy, n_steps)[:,0],
    )).T  

    Wx = jnp.stack((
        wx1 * make_exp_decay_weights(lam_Wx1, n_steps-1)[:,0],
        wx2 * make_exp_decay_weights(lam_Wx2, n_steps-1)[:,0],
    )).T  
    
    J, X_next, Y_hat = estimate_objective(
        f      = rk4(pendulum_dynamics, dt),
        g      = pendulum_output2,  # has output of dim 2 

        T      = jnp.arange( n_steps ) * dt,
        U      = U_excitation,
        Y      = Y_measurement, 
        Wx     = Wx,
        Wy     = Wy, 
        X      = X,
        theta  = theta,
    )
    
    return J.astype(jnp.float32), X_next.astype(jnp.float32), Y_hat.astype(jnp.float32)

def build_pendulum_model_estimate_objective2_tf_function(wy1, wy2, wx1, wx2, max_time, dt):
    n_steps = get_n_steps(max_time, dt)
    
    tf_fn = tf.function(
        jax2tf.convert(
            partial(pendulum_model_estimate_objective2, wy1=wy1, wy2=wy2, wx1=wx1, wx2=wx2, max_time=max_time, dt=dt)
        ), autograph=False,
        input_signature=[
            tf.TensorSpec(shape=(n_steps, 2), dtype=tf.float32, name='Y_measurement'),
            tf.TensorSpec(shape=(n_steps, 2), dtype=tf.float32, name='X'),
            tf.TensorSpec(shape=(3, ),        dtype=tf.float32, name='parameters'),
            tf.TensorSpec(shape=(   ),        dtype=tf.float32, name='lam'),
        ],
        reduce_retracing=True,
    )
    
    return tf_fn




def pendulum_estimation_jax(Y_measurement, max_time : float = 3.0, dt : float  = 0.01):
    
    n_steps = get_n_steps(max_time, dt)
    U_excitation = jnp.zeros( ( n_steps, 1) )
    
    wx = jnp.array([ 10.0, 10.0 ]) # high trust into the correctness of the model equations
    wy = jnp.array([ 0.1,       ]) # less trust into the correct measurement of the output as it is subjected to noise

    Wx, Wy = make_const_weights(wx, wy, n_steps)

    X_hat, theta_hat, Y_hat, J_star, res = estimate(
        
        f = rk4(pendulum_dynamics, dt),
        g = pendulum_output, 
        T = jnp.arange( n_steps ) * dt,
        U = U_excitation,
        Y = Y_measurement, 
        Wx = Wx, 
        Wy = Wy, 
        X0=jnp.zeros(( n_steps, 2 )), 
        theta0=jnp.array([4.01, 0.0, 0.0])
    )    
    
    return X_hat.astype(jnp.float32), theta_hat.astype(jnp.float32), Y_hat.astype(jnp.float32), J_star.astype(jnp.float32)

def build_pendulum_estimation_tf_function(max_time : float, dt : float):
    n_steps = get_n_steps(max_time, dt)

    tf_fn = tf.function(
        jax2tf.convert(
            partial(pendulum_estimation_jax, max_time=max_time, dt=dt)
        ), autograph=False,
        input_signature=[
            tf.TensorSpec(shape=(n_steps, 1), dtype=tf.float32, name='Y_measurement'),
        ],    
    )
    
    return tf_fn

#
# parameter identification (without state estimation)
#

def pendulum_parameter_identification_jax(Y_measurement, lam_Wy = 3.0, max_time : float = 3.0, dt : float = 0.01):
    
    n_steps = get_n_steps(max_time, dt)
    U_excitation = jnp.zeros( ( n_steps, 1) )

    #Wy = jnp.exp( -jnp.linspace(0, 1.0, n_steps) * lam_Wy ).reshape( (n_steps,1) )
    Wy = make_exp_decay_weights(lam_Wy, n_steps)

    theta_hat, x0_hat, X_hat, Y_hat, J_star, res = identify(
        
        f      = rk4(pendulum_dynamics, dt),
        g      = pendulum_output, 
        T      = jnp.arange( n_steps ) * dt,
        U      = U_excitation,
        Y      = Y_measurement, 
        Wy     = Wy, 
        x0     = jnp.array([ 0.0, 0.0 ]),
        theta0 = jnp.array([3.9, 0.0, 0.0])
    )    
    
    return tuple([ x.astype(jnp.float32) for x in [theta_hat, x0_hat, X_hat, Y_hat, J_star] ])

def build_parameter_identification_tf_function(max_time : float, dt : float):
    n_steps = get_n_steps(max_time, dt)

    tf_fn = tf.function(
        jax2tf.convert(
            partial(pendulum_parameter_identification_jax, lam_Wy = 3.0, max_time=max_time, dt=dt)
        ), autograph=False,
        input_signature=[
            tf.TensorSpec(shape=(n_steps, 1), dtype=tf.float32, name='Y_measurement'),
        ],    
    )
    
    return tf_fn


#
# Pendulum parameter estimation
# cost fn only for use in ML models
#

def pendulum_model_fit_cost( parameters, initial_states, Y_measurement, max_time : float, dt : float ):

    T, X, Y, U_excitation = simulate_pendulum( parameters, initial_states, max_time, dt  )
    J = jnp.mean( (Y_measurement - Y)**2 )
    
    return J.astype(jnp.float32), X.astype(jnp.float32), Y.astype(jnp.float32)

def build_pendulum_fit_cost_tf_function(max_time, dt):
    n_steps = get_n_steps(max_time, dt)
    
    print('build_pendulum_fit_cost_tf_function, n_steps=', n_steps)

    tf_fn = tf.function(
        jax2tf.convert(
            partial(pendulum_model_fit_cost, max_time=max_time, dt=dt)
        ), autograph=False,
        input_signature=[
            tf.TensorSpec(shape=(3, ),        dtype=tf.float32, name='parameters'),
            tf.TensorSpec(shape=(2, ),        dtype=tf.float32, name='initial_states'),
            tf.TensorSpec(shape=(n_steps, 1), dtype=tf.float32, name='Y_measurement'),
        ],
    )
    
    return tf_fn


#
# Tests
# 

# State estimation

def test_pendulum_model_estimate_objective():
    
    # set-up
    nominal_parameters = jnp.array( [4.0, 0.0, 0.0] )
    nominal_initial_states = jnp.array([ jnp.deg2rad(70.0), jnp.deg2rad(-10.0), ])
    max_time=3.0
    dt=0.01
    wy, wx1, wx2 = 1.0, 100.0, 100.0

    T, X_gt, Y, U_excitation = simulate_pendulum( nominal_parameters, nominal_initial_states, max_time=max_time, dt=dt )

    # act / assert
    J = pendulum_model_estimate_objective(Y, X_gt, nominal_parameters, 3.0, wy, wx1, wx2, max_time=max_time, dt=dt)
    assert J < 0.0001
    
    # set-up
    fn_tf = build_pendulum_model_estimate_objective_tf_function(wy, wx1, wx2, max_time, dt)
    
    # act / assert
    J_tf = fn_tf(
        tf.constant(Y, dtype=tf.float32), tf.constant(X_gt, dtype=tf.float32), 
        tf.constant(nominal_parameters, dtype=tf.float32), tf.constant(3.0, dtype=tf.float32)
    )
    
    assert (J - J_tf) < 0.0001
    

def test_tensorflow_conversion():
    
    # SET-UP
    nominal_parameters = np.array( [4.0, 0.0, 0.0] )
    nominal_initial_states = jnp.array([ jnp.deg2rad(70.0), jnp.deg2rad(-10.0), ])
    max_time=3.0
    dt=0.01
    
    pendulum_estimation_tf = build_pendulum_estimation_tf_function(max_time, dt)
    
    # generate test data
    T, X_gt, Y, U_excitation = simulate_pendulum( nominal_parameters, nominal_initial_states, max_time, dt  )

    # ACT: run jax function
    X_hat, theta_hat, Y_hat, J_star = pendulum_estimation_jax(Y, max_time, dt )
    
    # ASSERT Y_hat == Y
    tmp = np.array( Y_hat - Y )
    assert np.max( np.abs(tmp) ) < 0.001
    
    # ACT: run tensorflow function
    X_hat_tf, theta_hat_tf, Y_hat_tf, J_star_tf = pendulum_estimation_tf(Y)

    # ASSERT Y_hat == Y_hat_tf
    tmp = np.array( Y_hat - Y_hat_tf )
    assert np.max( np.abs(tmp) ) < 0.001


# Parameter identification

def test_pendulum_parameter_identification_jax():
    # set-up
    nominal_parameters = np.array( [4.0, 0.0, 0.0] )
    nominal_initial_states = jnp.array([ jnp.deg2rad(70.0), jnp.deg2rad(-10.0), ])
    max_time=10.0
    dt=0.01

    T, X_gt, Y, U_excitation = simulate_pendulum( nominal_parameters, nominal_initial_states, max_time=max_time, dt=dt )

    # act/assert
    theta_hat, x0_hat, X_hat, Y_hat, J_star = pendulum_parameter_identification_jax( Y, max_time=max_time, dt=dt )
    
    np.testing.assert_allclose(theta_hat, nominal_parameters, atol=0.001)
    assert J_star < 0.001 


# Cost function

def test_pendulum_model_fit_cost():
    # set-up
    nominal_parameters = np.array( [4.0, 0.0, 0.0] )
    nominal_initial_states = jnp.array([ jnp.deg2rad(70.0), jnp.deg2rad(-10.0), ])
    max_time=3.0
    dt=0.01
    
    T, X_gt, Y, U_excitation = simulate_pendulum( nominal_parameters, nominal_initial_states, max_time=max_time, dt=dt )
    
    # act/assert
    J, X_, Y_ = pendulum_model_fit_cost( nominal_parameters,      nominal_initial_states, Y, max_time=max_time, dt=dt )
    assert np.abs(J) < 0.0001
    
    # act/assert
    J, X_, Y_ = pendulum_model_fit_cost( nominal_parameters-0.1, nominal_initial_states, Y, max_time=max_time, dt=dt )
    assert np.abs(J) > 0.0001
    
def test_build_pendulum_fit_cost_tf_function():
    # set-up
    nominal_parameters = jnp.array( [4.0, 0.0, 0.0] )
    nominal_initial_states = jnp.array([ jnp.deg2rad(70.0), jnp.deg2rad(-10.0), ])
    max_time=3.0
    dt=0.01

    T, X_gt, Y, U_excitation = simulate_pendulum( nominal_parameters, nominal_initial_states, max_time=max_time, dt=dt )
    
    # act/assert
    pendulum_cost_tf_fn = build_pendulum_fit_cost_tf_function(max_time=max_time, dt=dt )
    J, X_, Y_ = pendulum_cost_tf_fn( nominal_parameters, nominal_initial_states, Y )
    assert np.abs(J) < 0.0001
    