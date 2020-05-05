#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: adrienbanse marinebranders
"""

import numpy as np
import matplotlib.pyplot as plt
from elevationMap import ElevationMap
import io

def pdf_e(e):
    return 1/(np.sqrt(var_e*2*np.pi)) * np.exp(-(e-mu_e)**2/(2*var_e))
    

# 1. parameters

with open('measures1D.txt', 'rb') as f:
    clean_lines = (line.replace(b'[',b',') for line in f)
    clean_lines = (line.replace(b']',b',') for line in clean_lines)
    clean_lines = (line.replace(b' ',b'') for line in clean_lines)
    [Y_t,POSITION_t] = np.genfromtxt(clean_lines, dtype=str, delimiter=',')   
    
Y_t         = np.array(Y_t[1:len(Y_t)-1],dtype=float)
POSITION_t  = np.array(POSITION_t[1:len(POSITION_t)-1],dtype=float)
Map         = ElevationMap("Ardennes.txt")

v_t     = 1.6   # speed
d_t     = 0.01  # time elapsed between measures
var_w   = 0.004 # variance of w_t for each t
mu_w    = 0     # mean of w_t for each t
var_e   = 16    # variance of e_t for each t
mu_e    = 0     # mean of e_t for each t
N       = 100   # number of particles
T       = 50    # final time

X_t     = np.zeros((N,T))
X_tilde = np.zeros((N,T))
m_x     = np.zeros(T)

# 2. sample from initial guess

X_t[:,0] = np.random.uniform(0,1,N)
m_x[0]   = 0.5

# 3. iterations

for t in range(T-1):
    
    # 3.1. state prediction
    w_t = np.random.normal(mu_w,np.sqrt(var_w),N)
    X_tilde[:,t+1] = (X_t[:,t] + v_t*d_t) + w_t
    
    # 3.2 weight update
    weight = np.zeros(N)
    for n in range(N):
        weight[n] = pdf_e(Y_t[t] - Map.h(float(X_tilde[n,t+1]))) # Y_t[t] = y_(t+1)
        
    weight = weight/np.sum(weight)
    
    X_t[:,t+1] = np.random.choice(X_tilde[:,t+1],size=N,p=weight)
    m_x[t+1] = np.sum(weight*X_tilde[:,t+1])

plt.plot(np.abs(POSITION_t-m_x))
#plt.plot(POSITION_t)
        
    
    
    
    
    
    
    
    
    
    
    