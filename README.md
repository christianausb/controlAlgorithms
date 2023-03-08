# controlAlgorithms
Algorithms at the boundary of control and machine learning

# Contents

## System identification
A (basic) implementation for the identification of non-linear systems implemented using the machine learning library JAX (https://github.com/google/jax). Herein, automatic differentiation of the system model and the through the ODE solver is used to enable gradient-based optimization approaches.

An example notebook describing the identification for a pendulum is provided https://nbviewer.org/github/christianausb/controlAlgorithms/blob/main/examples/sysident.ipynb

## State trajectory estimation and system identification

A routine for estimating the state trajectory and system parameters from input/output data and a prototype model is provided. The following example demonstrates the use for a pendulum system:

https://nbviewer.org/github/christianausb/controlAlgorithms/blob/main/examples/state_est_pendulum.ipynb


