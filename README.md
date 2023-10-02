# controlAlgorithms
Algorithms at the boundary of control and machine learning

![image](https://user-images.githubusercontent.com/4620523/236238763-343d0862-9265-464a-9208-35ea90b268fd.png)

# Contents

## System identification
A (basic) implementation for the identification of non-linear systems implemented using the machine learning library JAX (https://github.com/google/jax). Herein, automatic differentiation of the system model and the through the ODE solver is used to enable gradient-based optimization approaches.

An example notebook describing the identification for a pendulum is provided https://nbviewer.org/github/christianausb/controlAlgorithms/blob/main/examples/sysident.ipynb

## State trajectory estimation and system identification

A routine for estimating the state trajectory and system parameters from input/output data and a prototype model is provided. The following example demonstrates the use for a pendulum system:

https://nbviewer.org/github/christianausb/controlAlgorithms/blob/main/examples/state_est_pendulum.ipynb

## Pendulum motion estimation from video recordings

This experiment demonstrates how to combine state and parameter estimation with a deep neural autoencoder to estimate motion trajectories from video-recordings.

https://github.com/christianausb/controlAlgorithms/tree/main/examples/pendulum_finder

https://user-images.githubusercontent.com/4620523/223825323-2aa7c9f7-8d85-4b3c-aae0-8115737d95b7.mp4

## Trajectory optimization

An algorithm for the collocation method using optimzers from the jaxopt library is provided.

https://github.com/christianausb/controlAlgorithms/tree/main/examples/trajectory_optim_cart_pendulum.ipynb

https://github.com/christianausb/controlAlgorithms/assets/4620523/27e42c6d-ac39-4cbe-b7f3-5f5f7bb1b127



