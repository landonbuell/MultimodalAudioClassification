"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    SampleGeneration
    File:       sampleGenerator.py
    Classes:    SampleGenerator

    Author:     Landon Buell
    Date:       February 2025
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt

        #### CLASS DEFINITIONS ####

class SampleGenerationParameters:
    """ Structure to store parameters used to generate a waveform """

    class LowHighMeanVari:
        """ Stores Low, High, Mean, Variance for any parameter """
        
        def __init__(self,
                     low: float = 0.0,
                     high: float =1.0,
                     mean: float =0.0,
                     vari: float =1.0):
            """ Constructor """
            self.low    = low
            self.high   = high
            self.mean   = mean
            self.vari   = vari

    def __init__(self,
                 sampleRate: float = 44100.0):
        """ Constructor """
        self.sampleRate = sampleRate
        self.amp        = SampleGenerationParameters.LowHighMeanVari(1.0,10.0)
        self.freq       = SampleGenerationParameters.LowHighMeanVari(1.0,10.0)
        self.phase      = SampleGenerationParameters.LowHighMeanVari(0.0,2.0*np.pi)
        self.off        = SampleGenerationParameters.LowHighMeanVari(-0.1,+0.1)
        self.waveCount  = SampleGenerationParameters.LowHighMeanVari(1,16)
        self.inputAxis  = np.arange(0,2,1/sampleRate)

    def unpack(self) -> tuple:
        """ Unpack values """
        return (self.amp,self.freq,self.phase,self.off)

class GeneratedSample:
    """ Stores info about a generated sample """

    class ComponentWaveConfig:
        """ Stores Parameters for a single wave in a composite wave """
        
        def __init__(self):
            """ Constructor """
            self.amp    = 0
            self.freq   = 0
            self.phase  = 0
            self.off    = 0
            self.func   = None

        def unpack(self) -> tuple:
            """ Unpack values into a tuple """
            return (self.amp,self.freq,self.phase,self.off)

    def __init__(self,
                 waveform: np.ndarray,
                 sampleRate: float,
                 classIndex: int = -1,
                 params: list = None):
        """ Constructor """
        self.classInt       = classIndex
        self.waveform       = waveform
        self.sampleRate     = sampleRate
        self.params         = list()
        if (params is not None):
            self.params = params[:]

    # ACCESSORS

    @property
    def waveCount(self) -> int:
        """ Get the number of simple waves """
        return len(self.params)

    def __str__(self) -> str:
        """ Cast to string """
        s = "{0} @ {1} w/ {2} waves for class #{3}".format(
            self.__class__,
            hex(id(self)),
            self.waveCount,
            self.classInt)
        return s

    # PUBLIC INTERFACE

    def showWaveform(self) -> None:
        """ Plot the time-series representation of this waveform """
        titleText = "Generated Sample Class #{0}".format(self.classInt)
        
        plt.figure(figsize=(16,12))
        plt.title(titleText,size=24,weight='bold')
        plt.xlabel("Time [Sample Index]",size=16,weight='bold')
        plt.ylabel("Amplitude",size=16,weight='bold')

        plt.plot(self.waveform,color='blue')

        plt.vlines(0,ymin=np.min(self.waveform),ymax=np.max(self.waveform),color='black')
        plt.hlines(0,0,len(self.waveform),color='black')

        plt.grid()
        plt.tight_layout()

        plt.show()
        return None

    def exportBinaryToDisk(self,
                            outputFilePath: str,
                            sampleRate: float) -> bool:
        """ Export this sample to the provided output path in BINARY format """
        outputFileSize = len(self.waveform) + 1 # +1 for sample rate
        rawBytes = np.empty(shape=(outputFileSize,),dtype=self.waveform.dtype)
        rawBytes[0]     = sampleRate
        rawBytes[1:]    = self.waveform[:]
        rawBytes.tofile(outputFilePath)
        return True

    # PRIVATE METHODS

        
        

