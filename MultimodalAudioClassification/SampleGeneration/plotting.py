"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    SampleGeneration
    File:       plotting.py
    Classes:    -

    Author:     Landon Buell
    Date:       March 2024
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt

        #### FUNCTION DEFINITIONS ####

def plotSignal(yData,title):
    """ Show Time-Series Signal """
    plt.figure(figsize=(12,8))
    plt.title(title,fontsize=32,fontweight='bold')
    plt.xlabel("Time",fontsize=24,fontweight='bold')
    plt.ylabel("Amplitude",fontsize=24,fontweight='bold')

    plt.plot(yData,label="Signal")

    plt.ylim([-1.1,+1.1])
    plt.grid()
    plt.tight_layout()
    plt.legend()

    plt.show()
    return None

def plotSignalvsTime(xData,yData,title):
    """ Show Time-Series Signal """
    plt.figure(figsize=(12,8))
    plt.title(title,fontsize=32,fontweight='bold')
    plt.xlabel("Time",fontsize=24,fontweight='bold')
    plt.ylabel("Amplitude",fontsize=24,fontweight='bold')

    plt.plot(xData,yData,label="Signal")

    plt.ylim([-1.1,+1.1])
    plt.grid()
    plt.tight_layout()
    plt.legend()

    plt.show()
    return None