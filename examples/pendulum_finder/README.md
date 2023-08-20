# Pendulum finder

https://user-images.githubusercontent.com/4620523/223825323-2aa7c9f7-8d85-4b3c-aae0-8115737d95b7.mp4

The ML-algorithm in this experiment takes a video recoding of pendulum-like physical system and estimates a motion trajectory including position and velocity estimates. 
Herein, the algorithm does not require a calibration step or other (manual) measures. Instead it uses physics-informed ML to automatically find a pendulum motion as the cause of the recorded image sequence. 
In more detail, it combines a deep convolutional autoencoder, a physical model of a general pendulum in form of an ordinary differential equation (ODE), and a state and parameter estimation approach. While training this model on the
 recording, the model learns the state trajectory and, further, the parameters of the pendulum ODE, which form a subset of the trainable parameters.

This set-up was inspired by https://youtu.be/WHhDgxkiR9c [1].

## Architecture 

![pendulum_left_right](https://user-images.githubusercontent.com/4620523/236234977-56c1b66c-50da-496c-ba75-3b51bc00fa39.png)

a) The traditional way of estimating the motion of a pendulum which involves a sensor attached to the pendulum and a model-based state estimator to refine the sensor readings. b) Motion estimation using high-dimensional data, i.e., video recordings.

![pendulum_autoencoder drawio (3)](https://user-images.githubusercontent.com/4620523/233862854-f6ceaee8-b944-44ef-a992-3151ffe86753.png)

The used autoencoder structure to enable model training without labels (unsupervised learning).

## Other examples

### zucchini

https://user-images.githubusercontent.com/4620523/223848317-058861d4-c85e-4072-9e56-b132c84695c4.mp4

### leaf

https://user-images.githubusercontent.com/4620523/223848454-581e0452-6f00-4c25-ad1a-0be816708260.mp4

## References

[1] Champion, Kathleen, et al. "Data-driven discovery of coordinates and governing equations." Proceedings of the National Academy of Sciences 116.45 (2019): 22445-22451.
