#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: adrienbanse marinebranders
"""

import numpy as np
import matplotlib.pyplot as plt
from elevationMap import ElevationMap
import random
import io
    
# 1. data and parameters

with open('measures1D.txt', 'rb') as f:
    clean_lines = (line.replace(b'[',b',') for line in f)
    clean_lines = (line.replace(b']',b',') for line in clean_lines)
    clean_lines = (line.replace(b' ',b'') for line in clean_lines)
    [Y_t,POSITION_t] = np.genfromtxt(clean_lines, dtype=str, delimiter=',')   
    
Y_t         = np.array(Y_t[1:len(Y_t)-1],dtype=float)               # true data
POSITION_t  = np.array(POSITION_t[1:len(POSITION_t)-1],dtype=float) # true data

Map         = ElevationMap("Ardennes.txt") # Map object --> h method

v_t         = 1.6   # speed
d_t         = 0.01  # time elapsed between measures
var_w       = 0.004 # variance of w_t for each t
mu_w        = 0     # mean of w_t for each t
var_e       = 16    # variance of e_t for each t
mu_e        = 0     # mean of e_t for each t
N           = 100   # number of particles
T           = 50    # final time

X_t         = np.zeros((N,T)) # recolt particles
X_tilde     = np.zeros((N,T)) # recolt predictions
m_x         = np.zeros(T)     # recolt expected values
weight      = np.zeros(N)     # recolt weights at each iteration

pdf_e       = lambda e: 1/np.sqrt(var_e*2*np.pi) * np.exp(-0.5*((e-mu_e)/np.sqrt(var_e))**2)
    
# 2. sample from initial guess

X_t[:,0] = np.random.uniform(0,1,N) # sampling from initial uniform distribution
m_x[0]   = 0.5                      # initial expected value

# 3. iterations

for t in range(T-1):
    
    # 3.1. state prediction
    w_t = np.random.normal(mu_w,np.sqrt(var_w),N)
    X_tilde[:,t+1] = (X_t[:,t] + v_t*d_t) + w_t
    
    # 3.2. weight update
    for n in range(N):
        weight[n] = pdf_e(Y_t[t+1] - Map.h(float(X_tilde[n,t+1]))) # Y_t[t] = y_(t+1)
                
    weight = weight/np.sum(weight)
    
    # 3.3. resampling from estimated pdf
    X_t[:,t+1]  = np.random.choice(X_tilde[:,t+1],size=N,p=weight)
    #m_x[t+1]    = np.sum(weight*X_tilde[:,t+1])
        
# Visualization
plt.figure(1)
for t in range(T):
    print(t)
    # Display particles at each time:  
    for i in range(N):
        plt.plot(t,X_t[i,t],'ro',markersize=1)
  
    # Display true x at each time:
    plt.plot(t,POSITION_t[t],'kx')
  
    # Compute and display sample mean for each time:
    x_mean=0
    for i in range(N):
        x_mean = x_mean + X_t[i,t]
  
    x_mean = x_mean / N
    plt.plot(t,x_mean,'rx')
    
plt.xlabel('t')
plt.ylabel('x_t^i, i=1,...,n')
plt.title('Sequential Monte Carlo experiment')
plt.show()

plt.figure(2)
plt.plot(np.linspace(0,1,len(Y_t)),Y_t)
plt.show()
    
    
    
    
    
    
    
    
    
    
    