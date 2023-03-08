# Pendulum finder

https://user-images.githubusercontent.com/4620523/223825323-2aa7c9f7-8d85-4b3c-aae0-8115737d95b7.mp4

The ML-algorithm in this experiment takes a video recoding of pendulum-like physical system and estimates motion trajectory including position and velocity estimates. 
Herein, the algorithm does not require a calibration step or other (manual) measures. Instead, it uses a deep convolutional autoencoder, a physical model of a general pendulum in form 
of an ordinary differential equation (ODE), and a state and parameter estimation approach. While training this physics-informed model on the
input recording, the model learns the state trajectory and, further, the parameters of the pendulum ODE.

## Architecture 

![pendulum_autoencoder drawio (1)](https://user-images.githubusercontent.com/4620523/223824080-464af2b7-f964-4ccb-b758-4b6aca9d6890.png)


