#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks as findPeaks


plt.style.use('ggplot')             # if want to use the default style, set 'classic'
plt.style.use('ggplot')             # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False

#### Defined Functions ####

def computeDists(x,y):
    dists       = np.zeros((len(y),len(x)))
    
    for i in range(len(y)):
        for j in range(len(x)):
            dists[i,j]  = (y[i]-x[j])**2
            
    return dists


def pltDistances(dists,xlab="X",ylab="Y",clrmap="viridis"):
    imgplt  = plt.figure()
    plt.imshow(dists,
               interpolation='nearest',
               cmap=clrmap)
    
    plt.gca().invert_yaxis()    # This is added so that the y axis start from bottom, instead of top
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid()
    plt.colorbar()
    
    return imgplt



def computeAcuCost(dists):
    acuCost     = np.zeros(dists.shape)
    acuCost[0,0]= dists[0,0]
                                                # Calculate the accumulated costs along the first row
    for j in range(1,dists.shape[1]):           # the running number starts from 1, not 0
        acuCost[0,j]    = dists[0,j]+acuCost[0,j-1]
        
                                                # Calculate the accumulated costs along the first column
    for i in range(1,dists.shape[0]):           # the running number starts from 1, not 0
        acuCost[i,0]    = dists[i,0]+acuCost[i-1,0]    
    
                                                # Calculate the accumulated costs from second column, row onwards    
    for i in range(1,dists.shape[0]):
        for j in range(1,dists.shape[1]):
            acuCost[i,j]    = min(acuCost[i-1,j-1],
                                  acuCost[i-1,j],
                                  acuCost[i,j-1])+dists[i,j]
            
    return acuCost


def pltCostAndPath(acuCost,path,xlab="X",ylab="Y",clrmap="viridis"):
    px      = [pt[0] for pt in path]
    py      = [pt[1] for pt in path]
    
    
    imgplt  = pltDistances(acuCost,
                           xlab=xlab,
                           ylab=ylab,
                           clrmap=clrmap)  
    plt.plot(px,py)
    
    return imgplt
    


def pltWarp(s1,s2,path,xlab="idx",ylab="Value"):
    imgplt      = plt.figure()
    
    for [idx1,idx2] in path:
        plt.plot([idx1,idx2],[s1[idx1],s2[idx2]],
                 color="C4",
                 linewidth=2)
    plt.plot(s1,
             'o-',
             color="C0",
             markersize=3)
    plt.plot(s2,
             's-',
             color="C1",
             markersize=2)
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    return imgplt

###############################################################################
#### Create the function to perform dynamic time warping (to be completed) ####

def doDTW(x,y,dists,acuCost):




    return path,cost

###############################################################################
#### Extract ECG segments (to be completed) ####

# Extract the ECG
# --------------





# Find the peaks in the ECG
# -------------------------





# Extract ECG segments
# --------------------




###############################################################################
#### Perform dynamic time warping on ECG segments (to be completed) ####


# Comparing the first and second ecg, calculate the accumulated cost


print('The accumulated cost between ecg 1 and 2 is %f' % )



# Comparing the second and third ecg, calculate the accumulated cost


print('The accumulated cost between ecg 2 and 3 is %f' % )


# Comparing the second and sixth ecg, calculate the accumulated cost


print('The accumulated cost between ecg 2 and 6 is %f' % )

plt.show()
