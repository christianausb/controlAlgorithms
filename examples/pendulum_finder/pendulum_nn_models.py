import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model

from uncontrolled_pendulum import *



def encoder_decoder_real_picture_96x160x3(picture_shape):
    """
        Build CNN models for encoder, decoder, and picture differences
    """

    # [ [ 5*2**n, 3*2**n ] for n in range(6) ]

    encoder_model = tf.keras.models.Sequential([
        # input picsize 96, 160, 3

        tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"), # 100, 160, 8
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # -1, 50, 80, 8]

        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # output shape -1, 25, 40, 16

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # output shape -1, 13, 20, 32

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"), # 12, 20, 64
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # output shape -1, 7, 10, 64

        tf.keras.layers.Flatten(input_shape=picture_shape),
        tf.keras.layers.Dense(128),
    ], name='encoder_model_4CNN_layers_128out')

    decoder_model = tf.keras.models.Sequential([
        # input 128
        tf.keras.layers.Dense( 6 * 10 * 64  ),
        tf.keras.layers.Reshape( ( 6, 10, 64 ) ),

        tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"), #  -1, 14, 20, 32
        tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same"), # -1, 28, 40, 16
        tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=2, activation="relu", padding="same"), # -1,112, 160, 3
        tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=2, activation="relu", padding="same"), # -1,56, 80, 8
        tf.keras.layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same") # 112, 160, 3

    ], name='decoder_model_4CNNT_layers_128out')
        

    # delta model (processes the difference between two successive pictures)
    image_diff_to_position_delta_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"), # 100, 160, 8
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # -1, 50, 80, 8]

        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # output shape -1, 25, 40, 16

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # output shape -1, 13, 20, 32

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"), # 12, 20, 64
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # output shape -1, 7, 10, 64        
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128/2, activation='relu'),
        tf.keras.layers.Dense(1),
    ], name='image_to_position_delta_model')

    return encoder_model, decoder_model, image_diff_to_position_delta_model



def build_position_models_em128():
    """
        models that add dense layers to the CNNs to allow non-linear functions to be learned 
        that yield the position/angle
    """

    embedding_to_position_model = tf.keras.models.Sequential([
        # input 128
        tf.keras.layers.Dense(128/8, activation='relu'),
        tf.keras.layers.Dense(128/16, activation='relu'),
        tf.keras.layers.Dense(1),
    ], name='embedding_to_position_model')

    position_to_embedding_model = tf.keras.models.Sequential([
        # input 1
        tf.keras.layers.Dense(128/16),
        tf.keras.layers.Dense(128/8, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
    ], name='position_to_embedding_model')

    return embedding_to_position_model, position_to_embedding_model




#
# General autoencoder
#

class SeqAutoencoder(Model):
    """
        autoencoder that does *not* use a physical model
    """
    def __init__(self, encoder_model, decoder_model, embedding_model, n_y_additional=1, regularization_factor=1.0, name=None):
        
        super(SeqAutoencoder, self).__init__()
        
        self.regularization_factor = regularization_factor
        
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.embedding_model = embedding_model
        self.n_y_additional = n_y_additional # number of additional outputs of embedding_model
        
      
    def call(self, x_img_seq):
        
        encoder_model = self.encoder_model
        decoder_model = self.decoder_model
        embedding_model = self.embedding_model
        
        def chain(x_img_seq):
            
            Y = encoder_model(x_img_seq)
            # y_ : (n_steps, 1)
                        
            # 
            Y_hat, J_star, y_additional = embedding_model(Y)
            
            #
            img_seq_dec = decoder_model( Y_hat )
            
            return (img_seq_dec, J_star, ) + y_additional
            
        n_y_additional = 1
        
        # map the model chain over batch axis
        tmp = tf.map_fn(
            chain, x_img_seq,
            dtype=(tf.float32, tf.float32, ) + ( tf.float32, )*self.n_y_additional  # need to specify datatype for multi-output mapping
        )
        img_seq_dec, J_star_, y_additional_ = tmp[0], tmp[1], tmp[2:] 
        
        self.add_loss( self.regularization_factor * tf.reduce_mean(J_star_) )
        
        return (img_seq_dec, J_star_, ) + y_additional_
    
    

#
#
#

def build_R_sq_tf_function(n_steps : int):
    """
        generate a tensorflow function:
        
        R_sq, dB_R_sq = eval_R_squared( y, y_hat, lam )
    """
    
    def body(y, y_hat, lam):
        
        R_sq, dB_R_sq = eval_R_squared( 
            y, 
            y_hat,
            make_exp_decay_weights( lam, y_hat.shape[0], dtype=jnp.float32 ).reshape(-1)
        )
        
        return R_sq, dB_R_sq


    tf_fn = tf.function(
        jax2tf.convert(
            body
        ), autograph=False,
        input_signature=[
            tf.TensorSpec(shape=(n_steps), dtype=tf.float32, name='y'),
            tf.TensorSpec(shape=(n_steps), dtype=tf.float32, name='y_hat'),
            tf.TensorSpec(shape=(   ),     dtype=tf.float32, name='lam'),
        ],
        reduce_retracing=True,
    )
    
    return tf_fn




class WeightController(Model):
    """
        Controller that manipulates a parameter during training
        Also records internal training metrics into a tensor variable.
    """
    def __init__(
        self, 
        lambda_init = 1.5,
        controlled_variable_reference = -0.20,
        k_i = 4.0,
        
        lambda_rate_max =  0.01,
        lambda_rate_min = -0.02,
        
        n_batch = 1,
        n_history_max = 8000,
        is_controller_active = True,
        name=None
    ):
        
        super(WeightController, self).__init__()
        
        self.controller_active = is_controller_active
        
        self.lambda_init = lambda_init
        self.n_batch = n_batch
        self.controlled_variable_reference = controlled_variable_reference
        self.k_i = k_i
        
        self.lambda_rate_max = lambda_rate_max
        self.lambda_rate_min = lambda_rate_min
        
        self.weight_state = tf.Variable(
            lambda_init * tf.ones( (n_batch) ),
            trainable=False
        )

        self.step_counter = tf.Variable(
            int(0),
            trainable=False,
            dtype=tf.int64 # need to be int64 (https://www.tensorflow.org/xla/known_issues#tfvariable_on_a_different_device)
        )
        
        #
        
        n_vars_to_record = 5
        
        #
        self.recording = tf.Variable(
            tf.zeros( (n_history_max, n_batch, n_vars_to_record) ),
            trainable=False
        )

    def call(self, controlled_variable):
        
        e = self.controlled_variable_reference - controlled_variable
        
        to_int = self.k_i * e
        
        to_int_with_rate_limit = tf.clip_by_value(to_int, self.lambda_rate_min, self.lambda_rate_max) # maximal change of lambda per iteration
        
        weight_state_next = tf.clip_by_value(
            self.weight_state + to_int_with_rate_limit,
            0.0,
            self.lambda_init
        )
        
        if self.controller_active:
            self.weight_state.assign(weight_state_next)
        
        # counter
        self.step_counter.assign_add( tf.constant(1, dtype=tf.int64) )
        
        # record
        to_record = tf.transpose( tf.stack(( 
            controlled_variable, 
            self.weight_state,
            to_int,
            to_int_with_rate_limit,
            e,
        ), axis=0 ) )
        
        self.recording.assign(
            tf.concat( (
                self.recording[1:, :, :],   # drop the oldest recorded value
                tf.expand_dims( to_record, axis=0 ) 
            ), axis=0 )
        )
        
        return self.weight_state
        
def test_weight_controller():

    wc = WeightController(
        lambda_init = 1.5,
        controlled_variable_reference = 0.1,
        k_i = 4.0,
        n_batch = 2,
    )
    v1 = [  tf.constant( wc( tf.constant([0.1 + 0.25, 0.1 ]) ) ) for i in range(20) ]
    v2 = [  tf.constant( wc( tf.constant([0.1 - 0.05, 0.1 ]) ) ) for i in range(20) ]
    np.vstack(( np.array(v1), np.array(v2) ))    





class PendulumModelAutoencoder(Model):
    def __init__(
        self, 

        encoder_model, decoder_model, 
        embedding_to_position_model, position_to_embedding_model, 
        image_diff_to_position_delta_model,

        lambda_ml,
        lambda_mv,
        lambda_pendulum_fit,
        lambda_stability,

        wy1, wy2, wx1, wx2,

        lambda_exp,
        
        max_time=3.0, dt=0.01,
        n_experiments = 1,

        is_controller_active = True,

        name=None,
    ):
        
        super(PendulumModelAutoencoder, self).__init__()

        n_steps = get_n_steps(max_time, dt)
        self.n_steps = n_steps
        
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model        
        self.embedding_to_position_model = embedding_to_position_model
        self.position_to_embedding_model = position_to_embedding_model
        self.image_diff_to_position_delta_model = image_diff_to_position_delta_model
                
        self.regularization_factor = 1.0
        self.internal_Y_measurement_level = 1.0
        self.internal_Y_measurement_variance = 0.2
        
        self.wy1 = wy1
        self.wy2 = wy2
        self.wx1 = wx1
        self.wx2 = wx2 

        self.lambda_ml = lambda_ml
        self.lambda_mv = lambda_mv
        self.lambda_pendulum_fit = lambda_pendulum_fit
        self.lambda_stability    = lambda_stability    

        self.lambda_exp = lambda_exp
        
        self.pendulum_estimation_objective = build_pendulum_model_estimate_objective2_tf_function(
            wy1, wy2, wx1, wx2,
            max_time, dt
        )

        self.tfRsq = build_R_sq_tf_function( n_steps )

        # pendulum parameters (theta) for each batch, respectively
        self.theta_0 = tf.Variable( # a
            4.0 * tf.ones( (n_experiments), dtype=tf.float32 ),
            trainable=True
        )
        self.theta_1 = tf.Variable( # friction
            0.0 * tf.ones( (n_experiments), dtype=tf.float32 ),
            trainable=True
        )
        self.theta_2 = tf.Variable( # y_ofs
            0.0 * tf.ones( (n_experiments), dtype=tf.float32 ),
            trainable=True
        )
        
        # estimated states (X) for each batch, respectively
        self.X_hat = tf.Variable(
            0.0 * tf.ones( (n_experiments, n_steps, 2), dtype=tf.float32 ),
            trainable=True
        )

        # autotune of weight parameters
        self.wc = WeightController(
            lambda_init = self.lambda_exp,
            controlled_variable_reference = -0.20,
            k_i = 4.0,
            
            lambda_rate_max =  0.01,
            lambda_rate_min = -0.01,
            
            n_batch = 1,
            n_history_max = 5000,
            is_controller_active = True,
            name='controller_for_decay_parameter'
        )

    def call(self, x_img_seq):
        
        def body(xxx):  
                
            x_img_seq, theta_hat, X_hat, lam = xxx

            # log
            tf.summary.scalar('lambda (weight) control', lam )

            #
            # auto encoder model
            #
            
            # encode path
            embedding = self.encoder_model(x_img_seq)
            Y_measured__ = self.embedding_to_position_model(embedding) # (n_steps, 1)

            # remove bias introduced due to regularization
            Y_measured = Y_measured__ - self.internal_Y_measurement_level
                                    
            # decode path
            embedding_ = self.position_to_embedding_model(Y_measured__)
            img_seq_dec = self.decoder_model(embedding_)
            
            # use difference between two pictures to compute an estimate of the movement velocity
            x_img_seq_diff = tf.experimental.numpy.diff(x_img_seq, axis=0)
            Y_dot_measured = self.image_diff_to_position_delta_model( x_img_seq_diff )
                        
            Y_dot_measured = tf.concat( # hold last sample 
                (
                    Y_dot_measured, 
                    tf.reshape( Y_dot_measured[-1,:], (1,1) )  ),  # last sample
                axis=0 
            )

            #
            # Pendulum state trajectory and parameter estimation
            #

            # the measured system outputs (angle and angular velocity)
            Y_m = tf.concat( (Y_measured, Y_dot_measured), axis=1 ) 
                        
            # eval cost function of the estimator
            J_pendulum_fit, X_next, Y_hat = self.pendulum_estimation_objective(
                Y_m, X_hat,
                theta_hat, 
                lam,
            )
            
            # put constraint on friction (constrains the learned system to be stable) 
            b = theta_hat[1] # should be >0 for a stable system
            J_stability = tf.maximum( -b, 0 )

            #
            # metrics
            #
                        
            R_sq, dB_R_sq = self.tfRsq(
                Y_m[:,0],   # phi measured by nn
                Y_hat[:,0], # phi estimated by physical model
                lam,
            )

           # tf.print('metrics: R^2 =', R_sq, ' 20*log10(R^2) =', dB_R_sq, 'weights: lambda =', lam )
            tf.summary.scalar('R_sq of phi', R_sq)
            
            #
            # regularization
            #
            
            # put constraint for desired mean of position measurement
            Y_measurement_level = tf.reduce_mean( Y_measured__ ) 
            J_measurement_level = (self.internal_Y_measurement_level - Y_measurement_level)**2
        
            # put constraint for desired variance of position measurement
            J_measurement_variance = ( self.internal_Y_measurement_variance - tf.math.reduce_variance(Y_measured__) )**2
    

            # build combined cost
            J_star = (
                self.lambda_ml   * J_measurement_level +
                self.lambda_mv   * J_measurement_variance +
                
                self.lambda_pendulum_fit   * J_pendulum_fit +
                self.lambda_stability      * J_stability
            )
                        
            return img_seq_dec, J_star, Y_measured, Y_dot_measured, J_pendulum_fit, dB_R_sq
            

        theta_hat = tf.stack( (self.theta_0, self.theta_1, self.theta_2), axis=1 )   

        lam = self.wc.weight_state 
                
        # map the model over the batch axis
        img_seq_dec, J_star_, Y_measured, Y_dot_measurement, J_pendulum_fit, dB_R_sq = tf.map_fn(
            body, (x_img_seq, theta_hat, self.X_hat, lam ),
            dtype=( tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32 ) # need to specify datatype for multi-output mapping
        )

        # update weight controller
        self.wc(dB_R_sq)

        # apply regularization cost
        self.add_loss(  tf.reduce_sum( J_star_) )

        return (
            img_seq_dec, J_star_, Y_measured, Y_dot_measurement, tf.constant( 0.0 ),
            self.X_hat, theta_hat, J_pendulum_fit # additional signals by this module
        )



def build_physically_informed_model2(
    lambda_ml = 1.0,
    lambda_mv = 1.0,
    lambda_pendulum_fit = 1.0,
    lambda_stability = 0.1,

    wy1 = 1.0, wy2 = 1.0, wx1 = 100.0, wx2 = 100.0,

    lambda_exp = 0.0,  # set this to 0.0 to deactivate the controller during training

    picture_shape = None,
    max_time=None, dt=None
):
    """
        function thats sets-up the entire model for detecting pendulums
    """

    # picture CNN models
    encoder_model, decoder_model, image_diff_to_position_delta_model = encoder_decoder_real_picture_96x160x3(picture_shape)

    # non-linear functions
    embedding_to_position_model, position_to_embedding_model = build_position_models_em128()

    # autoencoder with pendulum model
    pe_autoencoder = PendulumModelAutoencoder(

        encoder_model, decoder_model, 
        embedding_to_position_model, position_to_embedding_model, 
        image_diff_to_position_delta_model,

        lambda_ml,
        lambda_mv,
        lambda_pendulum_fit,
        lambda_stability,

        wy1,
        wy2,
        wx1,
        wx2,

        lambda_exp,

        max_time = max_time, dt = dt,
        n_experiments = 1,
        name='pe_autoencoder',
    )
    
    return pe_autoencoder

