"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    SampleGeneration
File:       WaveformGenerators.py

Author:     Landon Buell
Date:       January 2023
"""

        #### IMPORTS ####

import os
import sys
import enum
import wave

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as scisig
import scipy.io.wavfile as sciowav

        #### FUNCTION DEFINITIONS ####

def plotSignal(xData,yData,title):
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

        #### CLASS DEFINITIONS ####

class WaveformType(enum.IntEnum):
    """ Enumeration for Waveform Type """
    SINE        = 0
    SQUARE      = 1
    TRIANGLE    = 2
    SAWTOOTH    = 3

class SimpleWavesforms:
    """ Static Generate Simple Waveforms """

    def __init__(self,):
        """ Constructor """


    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    @staticmethod
    def getSineWave(x,f,amp=1,phase=0,offset=0,noise=None):
        """ Generate a Sine wave w/ required params """
        y = amp * np.sin(2*np.pi*f*x + phase) + offset
        if (noise is not None):
            y += noise
        return y

    @staticmethod
    def getSquareWave(x,f,amp=1,phase=0,offSet=0,noise=None):
        """ Generate a Sine wave w/ required params """
        y = amp * scisig.square(2*np.pi*f*x + phase) + offset
        if (noise is not None):
            y += noise
        return y

    @staticmethod
    def getTriangleWave(x,f,amp=1,phase=0,offSet=0,noise=None):
        """ Generate a Sine wave w/ required params """
        y = amp * scisig.sawtooth(2*np.pi*f*x + phase,width=0.5) + offset
        if (noise is not None):
            y += noise
        return y

    @staticmethod
    def getSawtoothWave(x,f,amp=1,phase=0,offSet=0,noise=None):
        """ Generate a Sine wave w/ required params """
        y = amp * scisig.sawtooth(2*np.pi*f*x + phase,width=1.0) + offset
        if (noise is not None):
            y += noise
        return y

class SimpleNoise:

    @staticmethod
    def getUniformNoise(low,high,numSamples=1,scale=1):
        """ Get samples from a uniform distrobution """
        return np.random.Generator.uniform(low,high,numSamples) * scale

    @staticmethod
    def getGaussianNoise(mu,sigma,numSamples,scale=1):
        """ Get samples from a gaussian distrobution """
        return np.random.Generator.normal(mu,sigma,numSamples)* scale

class DatasetGenerator:
    """ Generate a Collection of Samples """

    def __init__(self,
                 callback,
                 timeAxis,
                 name):
        """ Constructor """
        self._generator     = callback
        self._timeAxis      = timeAxis
        self._name          = name

        self._amplitude     = 1
        self._freqBounds    = [110,12000]
        self._phase         = 0
        self._offset        = 0

    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    def createSamples(self,numSamples,exportPath):
        """ Create a Bumber of samples """
        if (os.path.isdir(exportPath) == False):
            os.makedirs(exportPath)
        self.__createSamples(numSamples,exportPath)
        return self


        

    # Private Interface

    def __createSamples(self,numSamples,exportPath):
        """ Helper to creat a collection of samples """
        for ii in range(numSamples):
            fundamental = np.random.uniform(
                low=self._freqBounds[0],
                high=self._freqBounds[1],
                size=1).astype(np.float32)
            noiseVector = (np.random.random(size=len(self._timeAxis)) - 0.5)/100.0
            noiseVector = noiseVector.astype(np.float32)
            signal = self._generator.__call__(
                self._timeAxis,
                fundamental,
                self._amplitude,
                self._phase,
                self._offset)
            noisySignal = signal + noiseVector
            noisySignal /= np.max(np.abs(noisySignal))
            fileName = self.__getNameForSample(fundamental[0],"wav")
            #plotSignal(self._timeAxis,noisySignal,fileName)
            outpath = os.path.join(exportPath,fileName)
            self.__toWavFile(noisySignal,outpath)
        return self

    def __getNameForSample(self,freq,ext="wav"):
        """ Get the Name ofthe output file for the current sample to generate """
        f = str(np.round(freq,2))
        return "{0}{1}Hz.{2}".format(self._name,f,ext)

    def __toWavFile(self,data,outputPath):
        """ Export Waveform as Wav File """
        print("Exporting: {0}".format(outputPath))
        sciowav.write(outputPath,44100,data)
        return self

    def __toBinaryFile(self,data,outputPath):
        """ Export Waveform as Binary File """
        return self









