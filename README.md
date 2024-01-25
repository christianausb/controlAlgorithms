# controlAlgorithms
Algorithms at the boundary of control and machine learning in [JAX](https://github.com/google/jax).

This package contains routines for (feedback) control of dynamic systems and considers
- the identification of dynamic systems from input/output data (learning parameters of ordinary differential equations),
- state estimation from input/output data with optional parameter learning, and
- model-based trajectory optimization and planning of nonlinear systems under inequality constraints.

The advantage of using JAX as an underlying basis is the automatic differentiation feature. Thus, gradients no not need to be (but can) provided by the code the implements the system models. Further, the optimization library [jaxopt](https://jaxopt.github.io/stable/index.html) is used for solving non-linear optimization problems. The routines can be jit-compiled for improved runtime performance.

![image](https://user-images.githubusercontent.com/4620523/236238763-343d0862-9265-464a-9208-35ea90b268fd.png)
***<p style="text-align: center;">A typical control system that involves state estimation (optionally sensor fusion) and feedback control.</p>***

# Installation

    pip install jax-control-algorithms

Local development install - clone repository and run

    pip install -e .

# Contents

## System identification
A (basic) least squares-based implementation for the identification of non-linear systems implemented is provided. Herein, an ODE solver and the system model are embedded in a non-linear optimization process.

An example notebook describing the identification for a pendulum is provided https://nbviewer.org/github/christianausb/controlAlgorithms/blob/main/examples/sysident.ipynb

## State trajectory estimation and system identification

A routine for estimating the state trajectory and system parameters from input/output data and a prototype model is provided. 

The following example demonstrates the estimation of a pendulum system using video-only data:

https://nbviewer.org/github/christianausb/controlAlgorithms/blob/main/examples/state_est_pendulum.ipynb

## Pendulum motion estimation from video recordings

This experiment demonstrates how to combine state and parameter estimation with a deep neural autoencoder to estimate motion trajectories from video-recordings.

https://github.com/christianausb/controlAlgorithms/tree/main/examples/pendulum_finder

https://user-images.githubusercontent.com/4620523/223825323-2aa7c9f7-8d85-4b3c-aae0-8115737d95b7.mp4

## Trajectory optimization

A routine that implements a variant of the collocation method is provided. To ensure inequality constraints, the penality method is implemented.

https://github.com/christianausb/controlAlgorithms/tree/main/examples/trajectory_optim_cart_pendulum.ipynb

A solution for the pendulum-cart benchmark is shown below. Herein, the control task is to swing-up the pendulum by controlling the cart motion.

https://github.com/christianausb/controlAlgorithms/assets/4620523/27e42c6d-ac39-4cbe-b7f3-5f5f7bb1b127



