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
import scipy.signal as scisig

        #### CLASS DEFINITIONS ####

class SampleGenerationParameters:
    """ Structure to store parameters used to generate a waveform """

    class LowHighMeanVari:
        """ Stores Low, High, Mean, Variance for any parameter """
        
        def __init__(self,
                     low: float=0.0,
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
        self.amp        = SampleGenerationParameters.LowHighMeanVari(1.0,10.0)
        self.freq       = SampleGenerationParameters.LowHighMeanVari(1.0,10.0)
        self.phase      = SampleGenerationParameters.LowHighMeanVari(0.0,2.0*np.pi)
        self.off        = SampleGenerationParameters.LowHighMeanVari(-0.1,+0.1)
        self.waveCount  = SampleGenerationParameters.LowHighMeanVari(1,64)
        self.inputAxis  = np.arange(0,2,1) / sampleRate

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
            """ Unpack values """
            return (self.amp,self.freq,self.phase,self.off)

    def __init__(self,
                 waveform: np.ndarray,
                 classIndex = -1,
                 params: list = None):
        """ Constructor """
        self.classInt       = classIndex
        self.waveform       = waveform
        self.params         = list()
        if (params is not None):
            self.params = params[:]

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

        
        

