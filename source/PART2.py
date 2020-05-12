#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: adrienbanse marinebranders
"""

#########################
###### IMPORTATION ######
#########################


import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from elevationMap import ElevationMap
import io

#import matplotlib.cm as cm
#import scipy.interpolate as si

###############################
###### USEFULS FUNCTIONS ######
###############################

def writeResult(fileName, M):
    with open(fileName, "w") as f:
        for i in range(len(M)):
            for j in range(len(M)):
                f.write("%14.7e " % M[i, j])
            f.write("\n")
        print(" === iteration %6d : writing %s ===" % (i, fileName))


def readResult(fileName, size):
    with open(fileName, "r") as f:
        E = np.array(list(list(float(w) for w in f.readline().split()) for i in range(size)))
    return E


def getData(dim_X):
    if dim_X == 1:
        with open('measures1D.txt', 'rb') as f:
            clean_lines = (line.replace(b'[', b',') for line in f)
            clean_lines = (line.replace(b']', b',') for line in clean_lines)
            clean_lines = (line.replace(b' ', b'') for line in clean_lines)
            [Y_t, POSITION_t] = np.genfromtxt(clean_lines, dtype=str, delimiter=',')
            POSITION_t = np.array([POSITION_t[1:len(POSITION_t) - 1]], dtype=float)  # true data
    else:
        with open('measures2D.txt', 'rb') as f:
            clean_lines = (line.replace(b'[', b',') for line in f)
            clean_lines = (line.replace(b']', b',') for line in clean_lines)
            clean_lines = (line.replace(b' ', b'') for line in clean_lines)
            [Y_t, POSITION1_t, POSITION2_t] = np.genfromtxt(clean_lines, dtype=str, delimiter=',')
            POSITION_t = np.array([POSITION1_t[1:len(POSITION1_t) - 1], POSITION2_t[1:len(POSITION2_t) - 1]],
                                  dtype=float)  # true data

    Y_t = np.array(Y_t[1:len(Y_t) - 1], dtype=float)  # true data
    return Y_t, POSITION_t

###################
###### PLOTS ######
###################

def plot():

    fig, ax = plt.subplots(dim_X,1, sharex = True, squeeze=False)
    fig.suptitle('Sequential Monte Carlo experiment')

    for t in range(T):
        print(t)
        
        for i in range(dim_X):
            ax[i,0].grid(True)
            ax[i,0].set_ylabel('x%1d_t^i, i=1,...,n [m]' % i)
            
            # Display particles at each time:
            ax[i,0].plot(t * np.ones(N), X_t[:, t,i], 'bo', markersize=0.5)

            # Display true x at each time:
            ax[i, 0].plot(t, POSITION_t[i, t], 'g.')

            # Compute and display sample mean for each time:
            mean_i = np.average(X_t[:, t,i])
            ax[i,0].plot(t,mean_i , 'k.')

            # Display error between sample mean and true x for each time:
            ax[i,0].plot(t, np.abs(mean_i - POSITION_t[i,t]), 'r.')
            
    plt.xlabel('t [s]')

def plotMap():
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.grid()

    # Plot the topography from topo.txt
    n           = 1000
    grid        = readResult("topo.txt", n)
    topo        = ma.masked_invalid(grid)
    lons, lats  = np.linspace(0, 1, n), np.linspace(0, 1, n)
    cs          = ax.contourf(lons, lats, topo, 80, cmap=plt.get_cmap('terrain'), alpha=0.1)
    cbar        = fig.colorbar(cs, extend='both', shrink=0.5, orientation='horizontal')
    cbar.set_label("topography height in meters")
    
    mx = np.zeros((T, 2)) # recolt means

    for t in range(T):
        print(t)
        
        # Display particles at each time:
        plt.plot(X_t[:, t, 0], X_t[:, t, 1], 'bo', markersize=0.1)
        
        # Display true x at each time:
        plt.plot(POSITION_t[0, t], POSITION_t[1, t], 'g.')

        # Compute and display sample mean for each time:
        mx[t] = [np.average(X_t[:, t, 0]), np.average(X_t[:, t, 1])]
        #plt.plot(mx[t, 0], mx[t, 1], 'x', color=cm.cool(np.abs(t) / T), markersize=5)
        plt.plot(mx[t, 0], mx[t, 1], 'k.')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Particle filtering : map vizualisation')
        
    # Display interpolation of mean with cubic splines
    # fx = si.CubicSpline(np.arange(T),mx[:,0])
    # fy = si.CubicSpline(np.arange(T),mx[:,1])
    # tlin = np.linspace(0,T,1000)
    # plt.plot(fx(tlin),fy(tlin),'-')

#################################
###### DATA AND PARAMETERS ######
#################################
        
dim_X   = 2
d_t     = 0.01          # time elapsed between measures
var_e   = 16            # variance of e_t for each t
mu_e    = 0             # mean of e_t for each t
N       = 1000          # number of particles
T       = 50 * dim_X    # total time (50 in 1D and 100 in 2D)

if dim_X == 1:
    var_w   = 0.004                                 # variance of w_t for each t
    v_t     = 1.6                                   # speed
else:
    var_w   = 1e-5 * np.array([[10, 8], [8, 10]])   # variance of w_t for each t
    v_t     = np.array([-0.4, -0.3])                # speed

Y_t, POSITION_t = getData(dim_X)                    # Observations Y_t and true data POSITION_t
Map             = ElevationMap(path="Ardennes.txt") # Map object --> h method

X_t     = np.zeros((N, T + 1, dim_X), dtype=float)  # recolt particles
X_tilde = np.zeros((N, T + 1, dim_X), dtype=float)  # recolt predictions
weight  = np.zeros(N)                               # recolt weights at each iteration

############################
###### MATH FUNCTIONS ######
############################
        
pdf_e       = lambda e:     1 / np.sqrt((2 * np.pi) * np.sqrt(var_e)) * np.exp(-0.5 * (e - mu_e) * (e - mu_e) / var_e) # pdf of noise e_t

def cholesky(var_w):
    
    # Since var_w is 2x2, we can easily find L on paper
    
    A = np.sqrt(var_w[0, 0])
    B = var_w[0, 1] / A
    C = np.sqrt(var_w[1, 1] - B * B)
    
    L = np.array([[A, 0], [B, C]])
    
    return L

#############################
###### PARTICLE FILTER ######
#############################

# 1. Sampling from initial distribution (uniform distribution)

X_t[:, 0] = np.random.uniform(0, 1, (N, dim_X))

# 2. Iterations

for t in range(T - 1):

    # 2.1. State prediction
    
    X = np.random.randn(N, dim_X) # we sample N times from a standard normal distribution
    
    if dim_X==2 : # we use the Cholesky decomposition of var_w
        L = cholesky(var_w)
    else: # 1D : L is the standard deviation
        L = np.array([[np.sqrt(var_w)]])
        
    w_t = np.dot(X,L)

    X_tilde[:, t + 1] = (X_t[:, t] + v_t * d_t) + w_t

    # 2.2. Weight update
    
    for n in range(N):
        if dim_X == 1:
            weight[n] = pdf_e(Y_t[t + 1] - Map.h(float(X_tilde[n, t + 1, 0])))
        else:
            weight[n] = pdf_e(Y_t[t + 1] - Map.h(X_tilde[n, t + 1]))
    weight = weight / np.sum(weight)

    # 2.3. Resampling from estimated pdf
    
    idx = np.random.choice(np.arange(N), size=N, p=weight)
    X_t[:, t + 1] = X_tilde[idx, t + 1]
    
#######################
###### VISUALIZE ######
#######################

plot()
if dim_X==2: # additional plot if 2D
    plotMap()
plt.show()
