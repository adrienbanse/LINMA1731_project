#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: adrienbanse marinebranders
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from elevationMap import ElevationMap
import io

# 1. data and parameters
dim_X = 2

if dim_X == 1 :
    with open('measures1D.txt', 'rb') as f:
        clean_lines = (line.replace(b'[', b',') for line in f)
        clean_lines = (line.replace(b']', b',') for line in clean_lines)
        clean_lines = (line.replace(b' ', b'') for line in clean_lines)
        [Y_t, POSITION_t] = np.genfromtxt(clean_lines, dtype=str, delimiter=',')
        POSITION_t = np.array(POSITION_t[1:len(POSITION_t) - 1], dtype=float)  # true data
else :
    with open('measures2D.txt', 'rb') as f:
        clean_lines = (line.replace(b'[', b',') for line in f)
        clean_lines = (line.replace(b']', b',') for line in clean_lines)
        clean_lines = (line.replace(b' ', b'') for line in clean_lines)
        [Y_t, POSITION1_t,POSITION2_t] = np.genfromtxt(clean_lines, dtype=str, delimiter=',')
        POSITION_t = np.array([POSITION1_t[1:len(POSITION1_t) - 1],POSITION2_t[1:len(POSITION2_t) - 1]], dtype=float)  # true data

Y_t = np.array(Y_t[1:len(Y_t) - 1], dtype=float)  # true data

Map = ElevationMap(path="Ardennes.txt")  # Map object --> h method

d_t = 0.01  # time elapsed between measures
if dim_X == 1:
    var_w = 0.004  # variance of w_t for each t
    mu_w = 0  # mean of w_t for each t
    v_t = 1.6  # speed
else:
    var_w = 1e-5 * np.array([[10, 8], [8, 10]])  # variance of w_t for each t
    mu_w = np.array([0, 0])  # mean of w_t for each t
    v_t = np.array([-0.4, -0.3])  # speed
var_e = 16  # variance of e_t for each t
mu_e = 0  # mean of e_t for each t
N = 1000  # number of particles
T = 100  # final time

X_t = np.zeros((N, T + 1, dim_X), dtype=float)  # recolt particles
X_tilde = np.zeros((N, T + 1, dim_X), dtype=float)  # recolt predictions
m_x = np.zeros((T, dim_X))  # recolt expected values
weight = np.zeros(N)  # recolt weights at each iteration

pdf_e = lambda e: 1 / np.sqrt((2 * np.pi) * np.sqrt(var_e)) * np.exp(-0.5 * (e - mu_e) * (e - mu_e) / var_e)
# lambda e: 1/np.sqrt(var_e*2*np.pi) * np.exp(-0.5*((e-mu_e)/np.sqrt(var_e))**2)











fig,ax = plt.subplots(figsize=(12,12))

import numpy.ma as ma
n = 1000
grid = np.zeros((n,n))
for i in range(n):
    print(i)
    for j in range(n):
        grid[i,j] = Map.h([i/n,j/n])

topo=ma.masked_invalid(grid)
lons, lats = np.linspace(0,1,n),np.linspace(0,1,n)

print(topo.min()) # minimum elevation in our domain (meters)
print(topo.max()) # maximum elevation in our domain (meters)

# Plot topography
cs = ax.contourf(lons, lats, topo, 80, cmap=plt.get_cmap('terrain'),alpha=0.1)
cbar = fig.colorbar(cs, extend='both', shrink=0.5, orientation='horizontal')
cbar.set_label("topography height in meters")












# 2. sample from initial guess

X_t[:, 0] = np.random.uniform(0, 1, (N, dim_X))  # sampling from initial uniform distribution
# m_x[0]   = 0.5                      # initial expected value

A = np.sqrt(var_w[0, 0])
B = var_w[0, 1] / A
C = np.sqrt(var_w[1, 1] - B * B)
L = np.array([[A, 0], [B, C]])

# 3. iterations

for t in range(T-1):

    # 3.1. state prediction
    if dim_X == 1:
        w_t = np.random.normal(mu_w, np.sqrt(var_w), (N, dim_X))
    else:
        X = np.random.randn(N, dim_X)
        w_t = np.dot(X, L)
        w_t += np.outer(np.ones(N), mu_w)

    X_tilde[:, t + 1] = (X_t[:, t] + v_t * d_t) + w_t

    # 3.2. weight update
    for n in range(N):
        if dim_X == 1:
            weight[n] = pdf_e(Y_t[t + 1] - Map.h(float(X_tilde[n, t + 1, 0])))
        else:
            weight[n] = pdf_e(Y_t[t + 1] - Map.h(X_tilde[n, t + 1]))
    weight = weight / np.sum(weight)

    # 3.3. resampling from estimated pdf
    idx = np.random.choice(np.arange(N), size=N, p=weight)
    X_t[:, t + 1] = X_tilde[idx, t + 1]
    # m_x[t+1]    = np.sum(weight*X_tilde[:,t+1])

# Visualization
X = np.arange(0, 1, 0.0001)
Y = np.zeros(len(X))
for i in range(len(X)):
    Y[i] = Map.h(float(X[i]))
cmap = cm.get_cmap('YlOrRd',T)
coef = 1 / pdf_e(0)
rgba = cmap(pdf_e(Y_t[t] - X[i]))


plt.figure(1)
plt.grid()
for t in range(T):
    print(t)
    # Display particles at each time:
    prob = np.zeros((len(X)))
    for i in range(len(X)):
        prob[i] = pdf_e(Y_t[t] - Y[i])
    # plt.scatter((t+0.3)*np.ones(len(X)),X,c=coef*prob,s=5,cmap = cm.get_cmap('YlOrRd'))
    #for i in range(N) :
    plt.plot(X_t[:, t,0] , X_t[:, t,1], 'o' , color =cm.hot(np.abs(t)/T), markersize=0.2)
    # plt.plot(t * np.ones(N), X_tilde[:, t], 'go', markersize=1)
for t in range(T):
    print(t)
    # Display true x at each time:
    plt.plot(POSITION_t[0,t], POSITION_t[1,t],'>', color = cm.cool(np.abs(t)/T),markersize=3)

    # Compute and display sample mean for each time:
    # x_mean = np.average(X_t[:, t,0])

    plt.plot(np.average(X_t[:, t,0]), np.average(X_t[:, t,1]), 'x', color = cm.cool(np.abs(t)/T),markersize=5)
    #
    # # Display error between sample mean and true x for each time:
    # plt.plot(t, np.abs(x_mean - POSITION_t[t]), 'ro', markersize=1)
    # plt.plot(t, np.abs(Y_t[t] - Map.h(float(POSITION_t[t]))) / 32, 'yo', markersize=1)

plt.xlabel('t')
plt.ylabel('x_t^i, i=1,...,n')
plt.title('Sequential Monte Carlo experiment')

# plt.figure(2)
# # plt.plot(Y)
# # plt.show()
# plt.grid()
# for t in range(T):
#     print(t)
#     for i in range(N) :
#         plt.plot((t + 0.05), X_t[i, t,1], 'bo', markersize=1)
#     plt.plot(t, POSITION_t[1,t], 'kx')
#     x_mean = np.average(X_t[:, t, 1])
#     plt.plot(t, x_mean, 'bx')
#
#
# plt.xlabel('t')
# plt.ylabel('x_t^i, i=1,...,n')
# plt.title('Sequential Monte Carlo experiment')

plt.show()


# plt.figure(2)
# plt.plot(np.linspace(0,1,len(Y_t)),Y_t)
# plt.show()















