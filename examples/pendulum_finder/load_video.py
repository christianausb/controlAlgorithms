import tensorflow as tf
import cv2
import numpy as np

def load_video(path, is_convert_to_rgb=False):
    frames = []
    cap = cv2.VideoCapture( path )
    read = True
    
    while read:
        read, img = cap.read()
        if read:
            if is_convert_to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            
    return np.stack(frames, axis=0)
    
def load_dataset_single_video(vfile, target_size = [100, 640//4], is_remove_meanpic = True, dt = 1/60.0):
    
    video = tf.constant(load_video(vfile))
    video = tf.image.resize(video, target_size) / 255.0
    
    # add batch dimension yielding a dataset
    x_train = tf.reshape( video, ( 1, ) + video.shape )

    mean_picture = tf.reduce_mean( x_train, axis=1 )

    if is_remove_meanpic:
        x_train = x_train-mean_picture
        x_train = (x_train-tf.reduce_min(x_train)) / (tf.reduce_max(x_train)-tf.reduce_min(x_train))

    max_time = dt*video.shape[0]
    T_batch = np.arange(video.shape[0])*dt
    picture_shape = tuple(x_train.shape[-3:])
    
    return x_train, mean_picture, T_batch, picture_shape, max_time, dt


