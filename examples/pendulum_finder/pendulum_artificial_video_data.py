from draw_pendulum import *
from uncontrolled_pendulum import *
import jax.numpy as jnp


def make_picture_sequence(Y, length=1.0, r=2.5):

    angle = Y[:,0]

    draw_fn = partial(draw_pendulum, length=length, r=r, meters_to_pixel=20)
    canvas_sequence = animate_pendulum(Y, draw_fn)
    canvas_sequence_1ch = canvas_sequence[:,:,:,0]
    
    return canvas_sequence_1ch, canvas_sequence

def simulation_and_pictures( pars, max_time : float, dt : float ):
    # combines simulation and picture
    
    T, X_gt, Y, U_excitation = simulate_pendulum( pars['parameters'], pars['initial_states'], max_time=max_time, dt=dt )
    pic_seq_1ch, _ = make_picture_sequence( Y, length=pars['length'], r=pars['r'] )
    
    return T, pic_seq_1ch, Y


def generate_artificial_dataset_single_batch(max_time = 10.0, dt = 0.01):

    n_steps = get_n_steps(max_time, dt)

    freq_fac = 0.5

    pars_1 = {
        'r' : 2.5,
        'length' : 1.0,
        'parameters' :  np.array( [freq_fac*8.0, 0.0, 0.0] ),
        'initial_states' : jnp.array([ jnp.deg2rad(40.0), jnp.deg2rad(-10.0), ]),
    }
    pars_2 = {
        'r' : 2.5,
        'length' : 1.0,
        'parameters' :  np.array( [freq_fac*9.0, 0.0, 0.0] ),
        'initial_states' : jnp.array([ jnp.deg2rad(-30.0), jnp.deg2rad(+10.0), ]),
    }

    #pic_seq_1ch = make_sample_sequence( pars )

    T_batch, pic_seq_1ch_1, Y_1 = simulation_and_pictures( pars_1, max_time, dt )
    _,       pic_seq_1ch_2, Y_2 = simulation_and_pictures( pars_2, max_time, dt )

    #seq_batch = jnp.stack( ( pic_seq_1ch_1, pic_seq_1ch_2 ) )
    #Y_batch = jnp.stack( ( Y_1, Y_2 ) )

    seq_batch = jnp.stack( ( pic_seq_1ch_1,  ) )
    Y_batch = jnp.stack( ( Y_1,  ) )

    T_batch.shape, seq_batch.shape, Y_batch.shape


    # training dataset 
    assert T_batch.shape[0] == Y_batch.shape[1]

    x_train = tf.constant( seq_batch, dtype=tf.float32 )
    y_train = tf.constant( Y_batch, dtype=tf.float32 )

    picture_shape = tuple(x_train.shape[-2:])

    return x_train, y_train, T_batch, picture_shape


