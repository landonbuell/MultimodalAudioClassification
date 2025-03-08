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

class SampleGeneratorConfig:
    """ Structure to store configuration parameters for a sample generator """

    def __init__(self,
                 className: str,
                 classIndex: int,
                 drawLimit: int = 1024,
                 callback):
        """ Constructor """
        self.className: str     = className
        self.classIndex: int    = classIndex
        self.drawLimit: int     = 1024
        self.callback: function = None

class SampleGenerationParameters:
    """ Structure to store parameters used to generate a waveform """

    class LowHighMeanVari:
        """ Stores Low, High, Mean, Variance for any parameter """
        
        def __init__(self,
                     low=0.0,
                     high=1.0,
                     mean=0.0,
                     vari=1.0):
            """ Constructor """
            self._data = np.array([low,high,mean,vari],dtype=np.float32)

        @property
        def low(self) -> np.float32:
            """ Return the low """
            return self._data[0]

        @low.setter
        def low(self,low:np.float32) -> None:
            """ Set the low """
            self._data[0] = low
            return None

        @property
        def high(self) -> np.float32:
            """ Return the high """
            return self._data[1]

        @high.setter
        def high(self,high:np.float32) -> None:
            """ Set the high """
            self._data[1] = high
            return None

        @property
        def mean(self) -> np.float32:
            """ Return the low """
            return self._data[2]

        @mean.setter
        def mean(self,mean:np.float32) -> None:
            """ Set the low """
            self._data[2] = mean
            return None

        @property
        def vari(self) -> np.float32:
            """ Return the high """
            return self.variance[3]

        @vari.setter
        def vari(self,vari:np.float32) -> None:
            """ Set the variance """
            self._data[3] = vari
            return None

        def __getitem__(self,key: int) -> np.float32:
            """ Get the item at the underlying data """
            return self._data[key]

        def __setitem__(self,key: int, val: np.float32) -> None:
            """ Set the item at the underlying data """
            self._data[key] = val
            return None

    def __init__(self):
        """ Constructor """
        self.amp        = SampleGenerationParameters.LowHighMeanVari(1.0,10.0)
        self.freq       = SampleGenerationParameters.LowHighMeanVari(1.0,10.0)
        self.phase      = SampleGenerationParameters.LowHighMeanVari(0.0,2.0*np.pi)
        self.off        = SampleGenerationParameters.LowHighMeanVari(-1.0,+1.0)
        self.waveCount  = SampleGenerationParameters.LowHighMeanVari(1,64)
        self.inputAxis      = np.arange(0,2,1) / 44100.0

class GeneratedSample:
    """ Stores info about a generated sample """

    class Parameters:
        """ Stores Parameters for a single wave in a composite wave """
        
        def __init__(self):
            """ Constructor """
            self.amp    = 0
            self.freq   = 0
            self.phase  = 0
            self.off    = 0
            self.func   = None

    def __init__(self,
                 waveform: np.ndarray,
                 classIndex = -1,
                 params=None):
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

        
        

