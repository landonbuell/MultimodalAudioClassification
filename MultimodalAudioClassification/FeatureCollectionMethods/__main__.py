"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       __main__.py
    Classes:    -

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import scipy.io.wavfile as sciowav

import signalData
import analysisFrames

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":
    
    # Create the unit Tests
    #filePath = "C:\\Users\\lando\\Documents\\audioWav2\\TRUMPET.Cs6.025.mezzoforte.normal.wav"
    #(sampleRate,waveform) = sciowav.read(filePath) 
    sampleRate = 44100
    t = np.arange(0,int(1e5),1) / sampleRate
    waveformA = np.cos( 2 * np.pi * 880 * t ) 
    waveformB = np.cos( 2 * np.pi * 1760 * np.log(55 * t + 1e-8 ) * t)
    waveform = waveformA + waveformB

    signal = signalData.SignalData(sampleRate,-1,waveform)
    frameParams = analysisFrames.AnalysisFrameParameters.defaultFrameParams()

    madeMFCCs = signal.makeMelFrequencyCepstralCoeffs(16,frameParams)
    sys.exit(0)



