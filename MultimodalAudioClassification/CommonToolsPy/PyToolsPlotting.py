"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifier
Project:        CommonUtilitiesPy
File:           Plotting.py
 
Author:         Landon Buell
Date:           April 2022
"""

        #### IMPORTS ####

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

        #### CLASS DEFINITIONS ####

def spectrogram(spectrogram,timeAxis,freqAxis,title,savepath=None,show=True):   
    """
    Plot a Heat-Map Spectrogram
    """
    plt.figure(figsize=(16,12))
    plt.title(title,size=24,weight='bold')
    plt.xlabel("Frequency",size=20,weight='bold')
    plt.ylabel("Time Frame Index",size=20,weight='bold')

    # Plot the Stuff
    plt.pcolormesh(freqAxis,
                   timeAxis,
                   spectrogram,
                   cmap=plt.cm.viridis)

    # House Keeping
    plt.grid()
    if (savepath is not None):
        plt.savefig(savepath)
    if (show == True):
        plt.show()
    return None

def heatMap(image,title,savepath=None,show=True):   
    """
    Plot a Heat-Map Spectrogram
    """
    plt.figure(figsize=(16,12))
    plt.title(title,size=24,weight='bold')
    plt.xlabel("Time",size=20,weight='bold')
    plt.ylabel("Frequency",size=20,weight='bold')

    # Plot the Stuff
    plt.imshow(image,cmap=plt.cm.binary)

    # House Keeping
    plt.grid()
    if (savepath is not None):
        plt.savefig(savepath)
    if (show == True):
        plt.show()
    return None

def plotSignal(xData,yData,title,savepath=None,show=True):
    """
    Plot a 1D Signal 
    """
    plt.figure(figsize=(16,12))
    plt.title(title,size=24,weight='bold')
    plt.xlabel("Domain",size=20,weight='bold')
    plt.ylabel("Amplitude",size=20,weight='bold')

    # Plot the Stuff
    plt.plot(xData,yData,color='blue')

    # House Keeping
    plt.grid()
    if (savepath is not None):
        plt.savefig(savepath)
    if (show == True):
        plt.show()
    return None
