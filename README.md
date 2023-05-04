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



