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
    t = np.arange(88200,dtype=np.float32) / 44100.0
    waveform = np.sin(2*np.pi*t*440)
    return waveform

def getSine880HzSignal() -> np.ndarray:
    """ Return Signal w/ 880 Hz Sine waveform """
    t = np.arange(88200,dtype=np.float32) / 44100.0
    waveform = np.sin(2*np.pi*t*880)
    return waveform

def getSine440Hz880HzSignal() -> np.ndarray:
    """ Return Signal w/ 440Hz & 880 Hz Sine waveform """
    t = np.arange(88200,dtype=np.float32) / 44100.0
    waveform = np.sin(2*np.pi*t*880) + np.sin(2*np.pi*t*440)
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

def getNormalWhiteNoise() -> np.ndarray:
    """ Return Signal w/ normalized white noise waveform """
    waveform = np.random.normal(0,1,size=88200) / 44100.0
    return waveform.astype(np.float32)

def getUniformWhiteNoise() -> np.ndarray:
    """ Return signal w/ uniform white noise """
    waveform = np.random.uniform(low=-1,high=1,size=88200)
    return waveform.astype(np.float32)
