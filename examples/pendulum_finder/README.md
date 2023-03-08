# Pendulum finder



The ML-algorithm in this experiment takes a video recoding of pendulum-like physical system and estimates motion trajectory including position and velocity estimates. 
Herein, the algorithm does not require a calibration step or other (manual) measures. Instead, it uses a deep convolutional autoencoder, a physical model of a general pendulum in form 
of an ordinary differential equation (ODE), and a state and parameter estimation approach. While training this physics-informed model on the
input recording, the model learns the state trajectory and, further, the parameters of the pendulum ODE.

## Architecture 



