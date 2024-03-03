"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    SampleGeneration
    File:       dummyWaveforms.py
    Classes:    

    Author:     Landon Buell
    Date:       March 2024
"""

        #### IMPORTS ####

import numpy as np

        #### FUNCTIONS DEFINITIONS ####

def getSine440HzSignal() -> np.ndarray:
    """ Return Signal w/ 440Hz Sine waveform """
    t = np.arange(88200,dtype=np.float32)
    waveform = np.sin(2*np.pi*t*440)
    return waveform

def getSine880HzSignal() -> np.ndarray:
    """ Return Signal w/ 880 Hz Sine waveform """
    t = np.arange(88200,dtype=np.float32)
    waveform = np.sin(2*np.pi*t*880)
    return waveform

def getNormalWhiteNoise() -> np.ndarray:
    """ Return Signal w/ normalized white noise waveform """
    waveform = np.random.random(size=88200)
    waveform /= np.max(np.abs(waveform))
    return waveform

def getConstZeroSignal() -> np.ndarray:
    """ Return Signal w/ all zero waveform """
    waveform = np.zeros(shape=(88200,),dtype=np.float32)
    return waveform

def getConstOneSignal() -> np.ndarray:
    """ Return Signal w/ all 1's waveform """
    waveform = np.zeros(shape=(88200,),dtype=np.float32) + 1
    return waveform

def getLinearRampSignal() -> np.ndarray:
    """ Return Signal w/ increasing waveform """
    waveform = np.arange(88200,dtype=np.float32)
    return waveform

def getUniformNoise(numSamples: int) -> np.ndarray:
    """ Get Array of Uniform Random Noise """
    y = (np.random.random(numSamples) - 0.5) / 100.0
    return y.astype(np.float32)