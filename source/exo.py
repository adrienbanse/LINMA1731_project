# -*- coding: utf-8 -*-

# sequential_monte_carlo_simple_04PA_*.py - Started Mon 03 Mar 2020
#

#   This Python script implements a Sequential Monte Carlo (SMC) method for
#   a simple nonlinear dynamical system.
#
#   In the *_incomplete.py version, the goal is to fill out the parts marked by "HIDDEN".
#
#   Reference: Computational Methods in Statistics, Anuj Srivastava, August
#   24, 2009, http://stat.fsu.edu/~anuj/pdf/classes/CompStatII10/BOOK.pdf
#   See in particular problem 3 of section 10.5.
#   See also the slides on Nonlinear Filtering available on Moodle.


# ** Let us define the dynamical system using the following model:
# x_{t+1} = F(x_t) + Gamma u_t
# y_t = G(x_t) + w_t.
# where x_t in R^m, y_t in R^p,
# x_0 ~ N(mu_x,Sigma_x), u_t ~ N(mu_u,Sigma_u), w_t ~ N(mu_w,Sigma_w)

import numpy as np
import random  # for random.choices
import matplotlib.pyplot as plt

t_f = 20  # final time. Sugg: 20
d_x = 1  # dimension of state space; must be 1 in this script
d_y = 1  # dimension of output space; must be 1 in this script
d_u = 1  # dimension of u; must be 1 in this script
a = .5  # a (used in F below) should be close to zero for stability
b = 0  # b (used in F below) should be close to zero, or zero to get a linear system
F = lambda x: a * x + b * x ** 3  # choice of the function F for the dynamical system
Gamma = 1
G = lambda x: x  # choice of output function G. Sugg: identity function
mu_x = 0  # see definition above
Sigma_x = np.array([[1]])  # The current version requires a 2d-array but it assumes a (1, 1) shape
mu_u = 0
Sigma_u = np.array([[1e-4]])
mu_w = 0
Sigma_w = np.array([[1e-0]])

sqrt_Sigma_x = np.sqrt(Sigma_x)
sqrt_Sigma_u = np.sqrt(Sigma_u)
sqrt_Sigma_w = np.sqrt(Sigma_w)

out_noise_pdf = lambda w: 1 / np.sqrt((2 * np.pi) ** d_y * np.abs(np.linalg.det(Sigma_w))) * np.exp(
    -.5 * (w - mu_w) @ np.linalg.inv(Sigma_w) @ (w - mu_w))  # pdf of the output noise w_t

# ** Simulation: Generate y_t, t=0,..,t_f, that will be used as the
# observations in the SMC algorithm.

x_true = np.zeros((d_x, t_f + 1))  # allocate memory
y_true = np.zeros((d_y, t_f + 1))  # allocate memory

x_true[:, 0] = mu_x + sqrt_Sigma_x * np.random.randn(d_x, 1)  # set true initial state
for t in range(t_f):
    u_true = mu_u + sqrt_Sigma_u * np.random.randn(d_u, 1)  # HIDDEN
    x_true[:, t + 1] = F(x_true[:, t]) + Gamma * u_true  # HIDDEN
    w_true = mu_w + sqrt_Sigma_w * np.random.randn(d_y, 1)  # HIDDEN
    y_true[:, t] = G(x_true[:, t]) + w_true  # HIDDEN

w_true = mu_w + sqrt_Sigma_w * np.random.randn(d_y, 1)  # noise on the output at final time t_f
y_true[:, t_f] = G(x_true[:, t_f]) + w_true  # output at final time t_f

# *** SEQUENTIAL MONTE CARLO METHOD ***

n = int(1e2)  # sample set size. Sugg: 1e2
X = np.zeros((d_x, n, t_f + 1))  # particles will be stored in X
Xtilde = np.zeros((d_x, n, t_f + 1))  # to store the predictions

# ** Generate initial sample set {x_0^i,...,x_0^n}:

t = 0
for i in range(n):
    X[:, i, t] = mu_x + sqrt_Sigma_x * np.random.randn(d_x, 1)  # we sample from the distribution of x_0  # HIDDEN

# ** Start loop on time:

for t in range(t_f):

    print(t)
    # ** Prediction

    for i in range(n):
        u = mu_u + sqrt_Sigma_u * np.random.randn(d_u, 1)  # HIDDEN
        Xtilde[:, i, t + 1] = F(X[:, i, t]) + Gamma * u  # HIDDEN

    # ** Update

    y = y_true[:, t + 1]  # y is the true output at time t+1

    weights = np.zeros(n)
    for i in range(n):
        weights[i] = out_noise_pdf(y - G(Xtilde[:, i, t + 1]))  # HIDDEN

    # Resample the particles according to the weights:
    ind_sample = random.choices(population=np.arange(n), weights=weights, k=n)

    for i in range(n):
        X[:, i, t + 1] = Xtilde[:, ind_sample[i], t + 1]

# end for t


# ** Visualization
plt.figure(1)
for t in range(t_f + 1):
    print(t)
    # HIDDEN[[
    # Display particles at each time:
    for i in range(n):
        plt.plot(t, X[0, i, t], 'ro', markersize=1)

    # Display true x at each time:
    plt.plot(t, x_true[0, t], 'kx')
    # Display true y at each time:
    plt.plot(t, y_true[0, t], 'k>')

    # Compute and display sample mean for each time:
    x_mean = np.zeros((d_x, 1))
    for i in range(n):
        x_mean = x_mean + X[:, i, t]

    x_mean = x_mean / n
    plt.plot(t, x_mean[0], 'rx')
    # HIDDEN]]
plt.xlabel('t')
plt.ylabel('x_t^i, i=1,...,n')
plt.title('Sequential Monte Carlo experiment')
plt.show()