import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import jax.numpy as jnp
from typing import List



def _add_traces(fig, T, X, names, is_angle, row_ofs=1):
    
    nr = X.shape[1]
    
    for i in range(nr):

        y_ = jnp.rad2deg( X[:,i] ) if is_angle[i] else X[:,i]

        fig.add_trace(
            go.Scatter(
                x=T, y=y_, name=names[i], 
            ),
            row=i+row_ofs, col=1
        )
        fig.update_xaxes(title_text="time [s]", row=i+row_ofs, col=1)
        fig.update_yaxes(title_text=names[i], row=i+row_ofs, col=1)
        
    return fig
    
def plot_states(
    T : jnp.ndarray,
    X : jnp.ndarray, 
    T_U  : jnp.ndarray = None, 
    U  : jnp.ndarray = None, 
    T_Y : jnp.ndarray = None, 
    Y : jnp.ndarray = None,
    state_names : List[str] = ['angle [degrees]', 'velocity [degrees]'], 
    is_angle : List[bool] = [True, True] 
):
    """
        Plot a time series of the given data. Herein, T is the vector describing
        the sample times and X contains the multi-dimensional time series data to
        show. Optionally, a system input U can be added, wherein T_U are the sampling
        times of the input trace.
    """

    nx = X.shape[1]
    nu = U.shape[1] if U is not None else 0
    ny = Y.shape[1] if Y is not None else 0

    assert len(state_names) == nx
    assert len(is_angle) == nx
    assert T.shape[0] == X.shape[0]
    
    fig = make_subplots(
        rows=nx + nu + ny,
        cols=1,
        shared_xaxes=True,
    )

    # plot states
    fig = _add_traces(fig, T, X, state_names, is_angle, row_ofs=1)
    
    # plot input
    T_ = T if T_U is None else T_U

    assert U.shape[1] == 1
    assert T_.shape[0] == U.shape[0]

    fig = _add_traces(fig, T_, U, ['system input'], [False], row_ofs=nx+1)

    # plot output
    T_ = T if T_Y is None else T_Y

    assert Y.shape[1] == 1
    assert T_.shape[0] == Y.shape[0]

    fig = _add_traces(fig, T_, Y, ['system output'], [False], row_ofs=nx+nu+1)
        
    fig.update_layout(height=600, width=800)
    fig.show()
    

def plot_output_comparison(
    T_Y : jnp.ndarray = None, 
    Y1 : jnp.ndarray = None,
    Y2 : jnp.ndarray = None,
    names : List[str] = ['measured angle [degrees]', 'simulated angle [degrees]'], 
    is_angle : List[bool] = [True, True] 
):
    """
        Plot a time series comparison of two output signals Y1 and Y2.
    """
    
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
    )
    
    # plot output
    assert Y1.shape[1] == 1
    fig = _add_traces(fig, T_Y, Y1, [ names[0] ], [ is_angle[0] ], row_ofs=1)
    
    assert Y2.shape[1] == 1
    fig = _add_traces(fig, T_Y, Y2, [ names[1] ], [ is_angle[1] ], row_ofs=1)

    fig.update_yaxes(title_text='system output, angle [degrees]', row=1, col=1)

    fig.update_layout(height=600, width=800)
    fig.show()
    
    
    
def plot_state_comparison(
    T : jnp.ndarray = None, 
    X1 : jnp.ndarray = None,
    X2 : jnp.ndarray = None,
    names : List[str] = ['angle [degrees]', 'anglular velocity [degrees/s]'], 
    is_angle : List[bool] = [True, True],
    left_label_postfix = ' (1)',
    right_label_postfix = ' (2)',
):
    """
        Plot a time series comparison of two state trajectories X1 and T2.
    """
    
    n_states = X1.shape[1]
    assert X2.shape[1] == n_states
    assert X1.shape[0] == X2.shape[0]
    
    fig = make_subplots(
        rows=n_states,
        cols=1,
        shared_xaxes=True,
    )
    
    for i in range(n_states):
        
        fig = _add_traces(fig, T, X1[:,i].reshape(-1,1), [ names[i] + left_label_postfix  ], [ is_angle[i] ], row_ofs=i+1)
        fig = _add_traces(fig, T, X2[:,i].reshape(-1,1), [ names[i] + right_label_postfix ], [ is_angle[i] ], row_ofs=i+1)
        
        fig.update_yaxes(title_text=names[i], row=i+1, col=1)

    fig.update_layout(height=600, width=800)
    fig.show()
    