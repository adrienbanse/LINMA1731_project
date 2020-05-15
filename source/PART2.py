#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project in Stochastic Processes : part II

@author:    adrienbanse 
            marinebranders
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

############################
###### MATH FUNCTIONS ######
############################

pdf_e = lambda e: 1 / np.sqrt((2 * np.pi) * np.sqrt(var_e)) * np.exp(
    -0.5 * (e - mu_e) * (e - mu_e) / var_e)  # pdf of noise e_t


def cholesky(var_w):
    # Since var_w is 2x2, we can easily find L on paper

    A = np.sqrt(var_w[0, 0])
    B = var_w[0, 1] / A
    C = np.sqrt(var_w[1, 1] - B * B)

    L = np.array([[A, 0], [B, C]])

    return L

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
            POSITION_t = np.array([POSITION_t[1:len(POSITION_t) - 1]], 
                                   dtype=float) # true data
    else:
        with open('measures2D.txt', 'rb') as f:
            clean_lines = (line.replace(b'[', b',') for line in f)
            clean_lines = (line.replace(b']', b',') for line in clean_lines)
            clean_lines = (line.replace(b' ', b'') for line in clean_lines)
            [Y_t, POSITION1_t, POSITION2_t] = np.genfromtxt(clean_lines, dtype=str, delimiter=',')
            POSITION_t = np.array([POSITION1_t[1:len(POSITION1_t) - 1], 
                                   POSITION2_t[1:len(POSITION2_t) - 1]],
                                   dtype=float) # true data

    Y_t = np.array(Y_t[1:len(Y_t) - 1], dtype=float) # true data
    
    return Y_t, POSITION_t

def simulateData(dim_X, e, w, X0):
    POSITION_t = np.zeros((dim_X, 50*dim_X))
    Y_t = np.zeros(50*dim_X)
    POSITION_t[:,0] = X0
    if dim_X == 1:
        Y_t[0] = Map.h(float(X0))
    else:
        Y_t[0] = Map.h(X0)

    if e :
        Y_t[0] += np.sqrt(var_e) * np.random.randn(1)

    for t in range (1, 50*dim_X) :
        POSITION_t[:,t] = POSITION_t[:,t-1] + d_t*v_t
        if w :
            if dim_X == 1:
                POSITION_t[:,t] += np.sqrt(var_w)*np.random.randn(1)
            else :
                POSITION_t[:,t] += cholesky(var_w) @ np.random.randn(2)

        if dim_X == 1:
            Y_t[t]= Map.h(float(POSITION_t[:,t]))
        else :
            Y_t[t] = Map.h(POSITION_t[:,t])

        if e:
            Y_t[t] += np.sqrt(var_e) * np.random.randn(1)
    return Y_t, POSITION_t

###################
###### PLOTS ######
###################

def plot():
    
    if dim_X>1:
        fig, ax = plt.subplots(dim_X+1,1, sharex = True, squeeze=False)
    else:
        fig, ax = plt.subplots(dim_X,1, sharex = True, squeeze=False)
    #fig.suptitle('Sequential Monte Carlo experiment')

    for i in range(dim_X):
        ax[i, 0].set_ylim(-0.2, 1.2)
        ax[i,0].grid(True)
        ax[i,0].set_ylabel('x%1d_t^i, i=1,...,n' % (i+1))
        #ax[i, 0].plot(21*np.ones(10), np.linspace(-0.2, 1.2,10), 'm')
        
        for t in range(T):
            print(t)
            
            # Display particles at each time:
            if t == 0 :
                ax[i,0].plot(t * np.ones(N), X_t[:, t,i], 'bo', markersize=0.5, label = "particles after resampling")
            else :
                ax[i, 0].plot(t * np.ones(N), X_t[:, t, i], 'bo', markersize=0.5)

            # # Display particles tilde at each time:
            # if t == 0:
            #     ax[i, 0].plot((t-0.25) * np.ones(N), X_tilde[:, t, i], 'o', markersize=1.5, color = 'yellow', label = "particles before resampling")
            # else :
            #     ax[i, 0].plot((t - 0.25) * np.ones(N), X_tilde[:, t, i], 'o', color = 'yellow', markersize=1.5)

            weight = np.zeros(N)
            for n in range(N):
                if dim_X == 1:
                    weight[n] = pdf_e(Y_t[t] - Map.h(float(X_tilde[n, t, 0])))
                else:
                    weight[n] = pdf_e(Y_t[t] - Map.h(X_tilde[n, t ]))
            #ax[i, 0].plot(t, max(weight), '.', color = 'darkorange')
            #ax[i, 0].plot(t , max(weight)/np.sum(weight), 'm.')

        mean = np.zeros(T)
        for t in range(T):
            mean[t] = np.average(X_t[:, t, i])

        # Display true x at each time:
        ax[i, 0].plot(np.arange(T), POSITION_t[i, :], 'gx', label = 'true position')

        # Compute and display sample mean for each time:
        ax[i,0].plot(np.arange(T),mean , 'kx', label = 'sample mean position')

        # Display error between sample mean and true x for each time:
        ax[i,0].plot(np.arange(T), np.abs(mean - POSITION_t[i]), 'r.', label='error')

        if i == 0 : ax[i, 0].legend(loc='upper right')

            
    if dim_X>1:
        ax[dim_X,0].grid(True)
        ax[dim_X,0].set_ylabel('norm of the error')
        ax[dim_X,0].set_ylim(0,0.6)
        
        for t in range(T):
            mx = [np.average(X_t[:, t, 0]), np.average(X_t[:, t, 1])]
            plt.plot(t, np.linalg.norm(mx-POSITION_t[:,t]), 'r.')
            
            
    plt.xlabel('t')

def plotMap():
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the topography from topo.txt
    n           = 1000
    grid        = readResult("topo.txt", n)
    topo        = ma.masked_invalid(grid)
    lons, lats  = np.linspace(0, 1, n), np.linspace(0, 1, n)
    cs          = ax.contourf(lons, lats, topo, 80, cmap=plt.get_cmap('terrain'), alpha=0.1)
    cbar        = fig.colorbar(cs, extend='both', shrink=0.5, orientation='horizontal')
    cbar.set_label("topography height in meters")
    
    mx = np.zeros((T, 2)) # recolt means
    #plt.ylim(-0, 1)
    #plt.xlim(-0, 1)
    #plt.grid()
    for t in range(T):
        print(t)
        
        # Display particles at each time:
        if t == 0 :  plt.plot(X_t[:, t, 0], X_t[:, t, 1], 'bo', markersize=0.1, label = 'particles (after resampling)')
        else : plt.plot(X_t[:, t, 0], X_t[:, t, 1], 'bo', markersize=0.1)

    for t in range(T):
        print(t)
        # Display true x at each time:
        if t == 0 : plt.plot(POSITION_t[0, t], POSITION_t[1, t], 'gx', label='true position')
        else : plt.plot(POSITION_t[0, t], POSITION_t[1, t], 'gx')

        # Compute and display sample mean for each time:
        mx[t] = [np.average(X_t[:, t, 0]), np.average(X_t[:, t, 1])]
        if t == 0: plt.plot(mx[t, 0], mx[t, 1], 'kx', label='sample mean position')
        else : plt.plot(mx[t, 0], mx[t, 1], 'kx')

        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.legend(loc="upper left")
        #plt.title('Particle filtering : map visualization')
        
    # Display interpolation of mean with cubic splines
    # fx = si.CubicSpline(np.arange(T),mx[:,0])
    # fy = si.CubicSpline(np.arange(T),mx[:,1])
    # tlin = np.linspace(0,T,1000)
    # plt.plot(fx(tlin),fy(tlin),'-')
    
#################################
###### DATA AND PARAMETERS ######
#################################

Map             = ElevationMap(path="Ardennes.txt") # Map object --> h method

dim_X   = 2
d_t     = 0.01          # time elapsed between measures
var_e   = 16            # variance of e_t for each t
mu_e    = 0             # mean of e_t for each t
N       = 2000           # number of particles
T       = 50 * dim_X    # total time (50 in 1D and 100 in 2D)

if dim_X == 1:
    var_w   = 0.004                                 # variance of w_t for each t
    v_t     = 1.6                                   # speed
else:
    var_w   = 1e-5 * np.array([[10, 8], [8, 10]])   # variance of w_t for each t
    v_t     = np.array([-0.4, -0.3])                # speed

Y_t, POSITION_t = getData(dim_X)                    # Observations Y_t and true data POSITION_t
#Y_t, POSITION_t = simulateData(dim_X, True, True,[0.95, 0.5])


X_t     = np.zeros((N, T + 1, dim_X), dtype=float)  # recolt particles
X_tilde = np.zeros((N, T + 1, dim_X), dtype=float)  # recolt predictions
weight  = np.zeros(N)                               # recolt weights at each iteration


#############################
###### PARTICLE FILTER ######
#############################

# Step 1 : Sampling from initial distribution (uniform distribution)

X_t[:, 0] = np.random.uniform(0, 1, (N, dim_X))

# Step 2 : Iterations

for t in range(T - 1):
    print(t)

    # Step 2.1 : State prediction
    
    X = np.random.randn(N, dim_X) # we sample N times from a standard normal distribution
    if dim_X==2 :   # 2D : we use the Cholesky decomposition of var_w
        L = cholesky(var_w)
    else:           # 1D : L is the standard deviation
        L = np.array([[np.sqrt(var_w)]])

    # Step 2.2 : Weight update
    
    w_t = np.einsum('ij,kj->ki', L, X)
    X_tilde[:, t + 1] = (X_t[:, t] + v_t * d_t) + w_t

    # Step 2.2. Weight update

    for n in range(N):
        if dim_X == 1:
            weight[n] = pdf_e(Y_t[t + 1] - Map.h(float(X_tilde[n, t + 1, 0])))
        else:
            weight[n] = pdf_e(Y_t[t + 1] - Map.h(X_tilde[n, t + 1]))
    weight = weight / np.sum(weight)

    # Step 2.3 : Resampling from estimated pdf
    
    idx = np.random.choice(np.arange(N), size=N, p=weight)
    X_t[:, t + 1] = X_tilde[idx, t + 1]
    
#######################
###### VISUALIZE ######
#######################

plot()
if dim_X==2: # additional plot if 2D
    plotMap()
#else :
    # plt.figure(4)
    # plt.grid()
    # h = np.zeros(T)
    # hApprox = np.zeros(T)
    # mean = np.zeros(T)
    # #plt.plot(21 * np.ones(10), np.linspace(-35, 35, 10), 'm')
    # plt.ylim(-35,35)
    # plt.grid()
    # for i in range(T):
    #     h[i] = Map.h(float(POSITION_t[0,i]))
    #     mean[i] = np.average(X_t[:, i, 0])
    #     hApprox[i] = Map.h(float(mean[i]))
    # plt.stem(np.arange(T), Y_t - h, use_line_collection=True, markerfmt= 'ko', linefmt='k' )
    # #plt.title('Difference between altitude mesures and true altitudes')
    # plt.grid()
    # plt.xlabel('t')
    # plt.ylabel('y_t - h(x1_t)')



    # plt.figure(5)
    # plt.plot(np.arange(T), h, 'g-')
    # plt.plot(np.arange(T), hApprox, 'k-')
    # plt.plot(np.arange(T), hApprox + (Y_t-h), 'r-')

    # plt.figure(6)
    # h = np.zeros(1000)
    # X = np.linspace(0,1,1000)
    # for i in range(1000):
    #     h[i] = Map.h(float(X[i]))
    # plt.plot(X, h, 'k-')
    # plt.plot(np.linspace(0,1,10), Map.h(float(POSITION_t[0,21]))*np.ones(10), 'm')
    # plt.grid()
    # plt.title('True altitudes')
    # plt.xlabel('x1_t')
    # plt.ylabel('h(x1_t)')

    ## WEIGHTS PLOT
    # plt.figure(2)
    # for t in range(T):
    #     print(t)
    #     weight = np.zeros(N)
    #     for n in range(N):
    #         if dim_X == 1:
    #             weight[n] = pdf_e(Y_t[t] - Map.h(float(X_tilde[n, t, 0])))
    #         else:
    #             weight[n] = pdf_e(Y_t[t] - Map.h(X_tilde[n, t]))
    #     if t == 0 :
    #         plt.plot(t, max(weight), '.', color = 'darkorange', label = 'maximum not normed weight')
    #         plt.plot(t , max(weight)/np.sum(weight), 'm.', label = 'maximum normed weight')
    #     else :
    #         plt.plot(t, max(weight), '.', color='darkorange')
    #         plt.plot(t, max(weight) / np.sum(weight), 'm.')
    #     if t>40 : print(max(weight))
    # plt.grid()
    # plt.title('Maximum weights')
    # plt.xlabel('t')
    # plt.legend(loc = 'upper left')



plt.show()


