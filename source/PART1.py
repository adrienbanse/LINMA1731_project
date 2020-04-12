#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:54:5
0 2020

@author: adrienbanse
"""

import numpy as np
import matplotlib.pyplot as plt

def plot():
    
    N_max = 20
    N = np.arange(1,N_max)
    b_MAP   = -1/(3*(N+4))
    b_ML    = np.zeros(N_max-1)
    MSE_MAP = (2*N+1)/(9*(N+4)**2)
    MSE_ML  = 2/(9*N)
    
    plt.plot(N,b_MAP,   'r.',   label='b_MAP'   )
    plt.plot(N,b_ML ,   'g.',   label='b_ML'    )
    plt.plot(N,MSE_MAP, 'rx',   label='MSE_MAP' )
    plt.plot(N,MSE_ML,  'gx',   label='MSE_ML'  )
    
    plt.xlabel("N") ; plt.ylabel("Values of bias/MSE")
    
    plt.legend()
    plt.show()


def estimators():
    
    # data extraction
    path1 = "Estimators1.txt"
    path2 = "Estimators2.txt"
    data1 = open(path1, "r").read()
    data1=data1.replace("[","") ; data1=data1.replace("]","") ; data1=data1.replace(" ","") ; Z1 = data1.split(",")
    data2 = open(path2, "r").read()
    data2=data2.replace("[","") ; data2=data2.replace("]","") ; data2=data2.replace(" ","") ; Z2 = data2.split(",")
    N = len(Z1)
    for i in range(0,N): Z1[i] = int(Z1[i]=='True')
    for i in range(0,N): Z2[i] = int(Z2[i]=='True')
    
    # MAP estimators
    theta_MAP_1 = (1+np.sum(Z1))/(N+4)
    theta_MAP_2 = (1+np.sum(Z2))/(N+4)
    
    # ML estimators
    theta_ML_1  = np.mean(Z1)
    theta_ML_2  = np.mean(Z2)
    
    # print results
    print("theta_MAP_1 : ",theta_MAP_1)
    print("theta_ML_1  : ",theta_ML_1 )
    print("---------------------------------")
    print("theta_MAP_2 : ",theta_MAP_2)
    print("theta_ML_2  : ",theta_ML_2 )