# -*- coding: utf-8 -*-
"""
Ground elevation map 

@author: cecile hautecoeur
"""
import numpy as np

###########
# HELPERS #
###########
def getIndex(point,dic,ind):
    i = dic.get(point,ind)
    if i == ind:
        dic[point] = ind; ind+=1
    return i, ind

def getArray(dic):
    L = [*dic]; L.sort()
    L = np.array(L, dtype=float)
    
    iL = L[0]; eL = L[-1]
    return (L-iL)/(eL-iL)

def findIndex(x,lis):
    index = -1
    prev = lis[0]
    for l in lis:
        if l>=x:
            if index != -1 and x-prev < l-x:
                return index
            return index+1
        prev = l
        index +=1
    return index-1 # should not arrive here
    
class ElevationMap:
    def __init__(self,path="testSmall.txt"):
        # read data
        data = open(path, "r").read()
        
        dataLines = data.split("\n")
        
        dicX, dicY, dicZ = {},{},{}
        
        indX, indY = 0, 0
        for i in dataLines:
            point = i.split(" ")
            if len(point)==3:
                iX, indX = getIndex(point[0],dicX,indX)
                iY, indY = getIndex(point[1],dicY,indY)
                    
                dicZ[(iX,iY)] = point[2]
        
        # sorted array of the points, rescaled between 0 and 1
        self.X = getArray(dicX) 
        self.Y = getArray(dicY)
        
        # array containing the altitude of each point
        self.dataPoints = np.zeros((len(self.X),len(self.Y)))
        for i in range(len(self.X)):
            for j in range(len(self.Y)):
                self.dataPoints[i,j] = float(dicZ.get((i,j)))        


    ###########
    # MAPPING #
    ###########
    
    # Function that returns the ground elevation at point x. 
    # x is either a float in [0,1] or a 2D vector in [0,1]x[0,1]
    def h(self,x):
        if type(x)==float or type(x)==int:
            if 0<=x and x<=1:
                iX = findIndex(x,self.X)
                return self.dataPoints[iX,0]
            else:
                return 1e5
            
            
        else:
            assert len(x)==2, "x must be either a float or a 2D vector"
            if 0<=x[0] and 0<=x[1] and x[0]<=1 and x[1]<=1:
                iX = findIndex(x[0],self.X)
                iY = findIndex(x[1],self.Y)
                return self.dataPoints[iX,iY]
            else:
                return 1e5
