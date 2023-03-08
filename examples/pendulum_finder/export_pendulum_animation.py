import matplotlib.pyplot as plt 
import numpy as np
import jax.numpy as jnp
import math
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from load_video import load_video

def draw_fig(t, fig, axs, T_batch, video, X_hat_):
    # plotting line
    t_end = T_batch[-1]
    i = jnp.argmin(jnp.abs(T_batch-t))
    i_max = T_batch.shape[0]
    i_ = i_max-i

    axs[0].imshow(video[i])
    axs[0].set_axis_off()

    axs[1].plot( T_batch, X_hat_[:,1].at[-i_:].set(math.nan), 'grey', label='velocity' )
    axs[1].plot( T_batch, X_hat_[:,0].at[-i_:].set(math.nan), 'r', label='position' )

    axs[1].set_ylim(jnp.min(X_hat_)*1.1, jnp.max(X_hat_)*1.1)
    axs[1].set_xlim(0.0, t_end)
    axs[1].set_xlabel('time [s]')
    axs[1].set_ylabel('[degrees]')
    axs[1].legend()


def init_fig(batch_index : int, T_batch, X_hat):
    
    t_end = T_batch[-1]

    X_hat_ = jnp.array(X_hat[batch_index])
    X_hat_ = jnp.stack((
        jnp.rad2deg(X_hat_[:,0]) * 2 - 1,
        jnp.rad2deg(X_hat_[:,1]) * 2 - 1,
    )).T

    # matplot subplot
    fig, axs = plt.subplots(2,1, dpi=200 )
    
    return fig, axs, X_hat_, t_end


def build_video(batch_index : int, vfile : str, T_batch, X_hat ):

    video = load_video(vfile, is_convert_to_rgb=True)

    fig, axs, X_hat_, t_end = init_fig(batch_index, T_batch, X_hat)

    # method to get frames
    def make_frame(t):

        # clear
        [ ax.clear() for ax in axs ]

        draw_fig(t, fig, axs, T_batch, video, X_hat_)

        # returning numpy image
        return mplfig_to_npimage(fig)

    # creating animation
    animation = VideoClip(make_frame, duration = t_end)
    
    return animation
