import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt 

from load_video import load_dataset_single_video, load_video
from pendulum_nn_models import *
from jax_control_algorithms.estimation import eval_R_squared

def normalize(x):
    return ( x - jnp.min(x) ) / ( jnp.max(x) - jnp.min(x) )

#
# auto training
#

def run_training_task(pdf_scenarios, task, logfolder : str, jit_compile : bool):

    # load video
    x_train, mean_picture, T_batch, picture_shape, max_time, dt = load_train_data(pdf_scenarios, task.scenario)
    
    # build the model
    pe_autoencoder = build_physically_informed_model2(
        lambda_ml = task.lambda_ml,
        lambda_mv = task.lambda_mv,
        lambda_pendulum_fit = task.lambda_pendulum_fit,
        lambda_stability = task.lambda_stability,

        wy1 = task.wy1, wy2 = task.wy2, wx1 = task.wx1, wx2 = task.wx2, 

        lambda_exp = task.lambda_exp,

        picture_shape=picture_shape, 
        max_time=max_time, dt=dt
    )
    
    # test
    tmp = pe_autoencoder.predict(x_train)

    # compile
    pe_autoencoder.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = task.learning_rate),
        loss = [ tf.keras.losses.MeanSquaredError(), None, None, None, None, ],
        jit_compile = jit_compile
    )

    # train in multiple stages
    n_ep_count = 0
    loss_history = np.array([]).reshape( (0) )
    model_fname_history = []
    for i_snapshot, n_epochs in enumerate(task.n_epochs):
        
        history = pe_autoencoder.fit( x_train, x_train, epochs=n_epochs, verbose=True )
        loss_history = np.concatenate(( loss_history, np.array( history.history['loss'] ) ))

        n_ep_count += n_epochs

        # save snapshot
        model_fname_relative = os.path.join( str(task.scenario), str(task.task_id), f'snapshot_{i_snapshot}/model' )
        model_fname_history.append(model_fname_relative)      
    
        # 
        model_fname = os.path.join( logfolder, model_fname_relative )
        print(f'writing {model_fname}')
        pe_autoencoder.save( model_fname )
        
    return {
        'model_fnames' : model_fname_history,
        'task_id' : task.task_id,
        'scenario' : task.scenario,
        'loss_history' : loss_history,
    }

def run_traing_tasks(pdf_scenarios, pdf_training_tasks, logfolder='trained_models/autorun', jit_compile=False):

    os.mkdir(logfolder)

    # run
    pdf_training_tasks.to_pickle(f'{logfolder}/pdf_training_tasks.pickle')
    pdf_scenarios.to_pickle(f'{logfolder}/pdf_scenarios.pickle')

    results_history = []
    for i, task in pdf_training_tasks.iterrows():

        print('-- task --')
        print(task)
        results = run_training_task(pdf_scenarios, task, logfolder, jit_compile)
        results_history.append(results)

        # save results
        pdf_results = pd.DataFrame(results_history) #.model_fnames.iloc[0]
        pdf_results.to_pickle(f'{logfolder}/pdf_results.pickle')


#
# eval
#


def plot_time_eval(pe_autoencoder, x_train, T_batch, dt, i_batch = 0):

    
    tmp = pe_autoencoder.predict(x_train, verbose=False)
    J_pendulum_fit = tmp[7][i_batch]

    print(f"theta=[ {float(pe_autoencoder.theta_0)}, {float(pe_autoencoder.theta_1)}, {float(pe_autoencoder.theta_2)} ]")
    print( f'J_pendulum_fit = {J_pendulum_fit}' )

    phi = tmp[2][i_batch]
    phi_dot = tmp[3][i_batch]
    
    phi_hat = pe_autoencoder.X_hat[i_batch, :, 0]
    phi_dot_hat = pe_autoencoder.X_hat[i_batch, :, 1]
    
    R_sq, dB_R_sq = eval_R_squared( 
        jnp.array(phi[:,0]), 
        jnp.array(phi_hat),
        make_exp_decay_weights( 1.2, phi_hat.shape[0] ).reshape(-1)
    )
    
    print('R^2 =', np.array(R_sq), ' 20*log10(R^2) =', np.array( dB_R_sq ) )
    
    plt.figure(dpi=100)
    plt.subplot(2,1,1)
    plt.plot( T_batch, phi, 'g', label='measured by nn' )
    plt.plot( T_batch, phi_hat, 'k', label='estimated' )
    plt.ylabel('angle [degrees]')
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot( T_batch[:-1], 1/dt*np.diff(np.array(phi)[:,0]), 'lightgrey', label='derivative of measured angle' )
    plt.plot( T_batch, phi_dot, 'g', label='measured by nn' )
    plt.plot( T_batch, phi_dot_hat, 'k', label='estimated' )
    plt.xlabel('time [s]')
    plt.ylabel('angular velocity [degrees/s]')
    plt.legend()
    


def load_train_data(pdf_scenarios, scenario):
    
    vfile = pdf_scenarios[ pdf_scenarios.scenario == scenario ]['vfile'].iloc[0]
    print(f'loading {vfile}')

    x_train, mean_picture, T_batch, picture_shape, max_time, dt = load_dataset_single_video(
        vfile, 
        target_size = [96, 160], 
        is_remove_meanpic = True, 
        dt = 1/60.0
    )
    
    return x_train, mean_picture, T_batch, picture_shape, max_time, dt 
    


class ExperimentalData:
    def __init__(self, logfolder='trained_models/autorun'):
        self.logfolder = logfolder
        self._pdf_results        = pd.read_pickle( os.path.join( logfolder, 'pdf_results.pickle') )
        self._pdf_training_tasks = pd.read_pickle( os.path.join( logfolder, 'pdf_training_tasks.pickle') )
        self._pdf_scenarios      = pd.read_pickle( os.path.join( logfolder, 'pdf_scenarios.pickle') )
        pass
        
    @property
    def pdf_results(self):
        return self._pdf_results
        
    @property
    def pdf_training_tasks(self):
        return self._pdf_training_tasks

    @property
    def pdf_scenarios(self):
        return self._pdf_scenarios
        
    def get_model_path(self, task, i_snapshot):

        result = self._pdf_results[ self._pdf_results.task_id == task.task_id ]
        model_fname = result.model_fnames.iloc[0][i_snapshot]

        return model_fname

    def get_result_from_task( self, task ):
        result = self._pdf_results[ self._pdf_results.task_id == task.task_id ].iloc[0]
        return result

def plot_loss_history(data, task):
    result = data.get_result_from_task(task) 

    plt.plot( np.log10(result.loss_history), 'k' )
    plt.xlabel('epochs')
    plt.ylabel('log10(loss)')
    plt.show()

def plot_full_training_history(data, task, pe_autoencoder=None):
    pe_autoencoder = load_model(data, task) if pe_autoencoder is None else pe_autoencoder
    loss_history = data.get_result_from_task(task).loss_history
    
    _plot_full_training_history(pe_autoencoder, loss_history)

def _plot_full_training_history(pe_autoencoder, loss_history):
    n_iter = pe_autoencoder.wc.step_counter.numpy().item()
    dB_R_sq_history = pe_autoencoder.wc.recording[-n_iter:, 0, 0].numpy()
    lam_history = pe_autoencoder.wc.recording[-n_iter:, 0, 1].numpy()

    f, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True,figsize=(8,6),constrained_layout=True)

    ax1.plot(np.where(dB_R_sq_history < -0.2, np.nan, dB_R_sq_history ), 'k', label='logarithmic fit value; zero means best fit')
    ax1.set_ylabel('20 * log10( R^2 )')
    ax1.legend()

    ax2.plot(lam_history, 'k', label='weight decay factor')
    ax2.set_ylabel('lambda')
    ax2.legend()

    ax3.plot( np.log10(loss_history), 'k', label='loss')
    ax3.set_ylabel('log10(loss)')
    ax3.set_xlabel('training iterations')


def plot_eval_task(data : ExperimentalData, task):
    
    result = data.get_result_from_task(task)

    x_train, mean_picture, T_batch, picture_shape, max_time, dt = load_train_data(data.pdf_scenarios, task.scenario)

    for i_sbapshot, model_fname in enumerate( result.model_fnames ):
        print(f'snapshot #{i_sbapshot}', model_fname)
        
        pe_autoencoder = tf.keras.models.load_model(os.path.join( data.logfolder, model_fname ) )

        plot_time_eval(pe_autoencoder, x_train, T_batch, dt)
        plt.show()


def load_model(data : ExperimentalData, task, i_snaphot : int = None ):
    result = data.get_result_from_task(task)    
    model_fname = result.model_fnames[-1] if i_snaphot is None else result.model_fnames[i_snaphot]
    pe_autoencoder = tf.keras.models.load_model(os.path.join( data.logfolder, model_fname ) )

    return pe_autoencoder


def load_train_data_and_model( data : ExperimentalData, task, i_snaphot : int = None ):

    pe_autoencoder = load_model(data, task, i_snaphot)
    x_train, mean_picture, T_batch, picture_shape, max_time, dt = load_train_data(data.pdf_scenarios, task.scenario)

    return pe_autoencoder, x_train, mean_picture, T_batch, picture_shape, max_time, dt


def plot_eval_task_single_snapshot( data : ExperimentalData, task, i_snaphot : int = None ):

    pe_autoencoder, x_train, mean_picture, T_batch, picture_shape, max_time, dt = load_train_data_and_model( data, i_snaphot )

    plot_time_eval(pe_autoencoder, x_train, T_batch, dt)
    plt.show()


def make_video_from_eval_task(data : ExperimentalData, task):

    from export_pendulum_animation import build_video

    result = data.get_result_from_task(task)

    x_train, mean_picture, T_batch, picture_shape, max_time, dt  = load_train_data(data.pdf_scenarios, task.scenario) # TODO: rename load_video!

    model_fname = result.model_fnames[-1]
    print(f'', model_fname)

    pe_autoencoder = tf.keras.models.load_model(os.path.join( data.logfolder, model_fname ) )

    vfile = data.pdf_scenarios[ data.pdf_scenarios.scenario == task.scenario ]['vfile'].iloc[0]

    print(f'exporting video from {vfile}')

    #vfile = 'pendulum_videos/ApfelKurz480p.mov'
    animation = build_video(0, vfile, T_batch, pe_autoencoder.X_hat)

    return animation