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

import sampleFile

        #### CLASS DEFINITIONS ####

class SampleGenerationParameters:
    """ Structure to store params for a simple waveform generator """

    def __init__(self):
        """ Constructor """
        self.amplitudeLow   = 1.0
        self.amplitudeHigh  = 10.0
        self.frequencyLow   = 1.0
        self.frequencyHigh  = 10.0
        self.phaseLow       = 0.0
        self.phaseHigh      = 2.0*np.pi
        self.offsetLow      = -0.1
        self.offsetHigh     = +0.1
        self.inputAxis      = np.arange(0,2,1) / 44100.0
        self.waveCountLow   = 1
        self.waveCountHigh  = 64


class SampleGeneratorCallbacks:
    """ 
        Static class of callbacks to generate samples 
        All signatures must obey:

        def callbackName(
            params: SampleGenerationParameters) -> sampleFile.GeneratedSampleFileIO
       
    """

    @staticmethod
    def cosineUniform(params: SampleGenerationParameters) -> np.ndarray:
        """ Cosine Wave(s) - Params from Uniform Distrobution """
        x = np.zeros(shape=(params.inputAxis.size,),dtype=np.float32)
        waveCount = np.random.uniform(low=params.waveCountLow,low=params.waveCountHigh,size=1)
        # Generate Parameters
        a = np.random.Generator.uniform(params.amplitudeLow,params.amplitudeHigh,size=waveCount)
        f = np.random.Generator.uniform(params.frequencyLow,params.frequencyHigh,size=waveCount)
        p = np.random.Generator.uniform(params.phaseLow,params.phaseHigh,size=waveCount)
        o = np.random.Generator.uniform(params.offsetLow,params.offsetHigh,size=waveCount)
        # Create the waveform:      
        for ii in range(waveCount):
            x += a[ii] * np.cos((2 * np.pi * params.inputAxis * f[ii]) + p[ii]) + o[ii]
        return x    

    @staticmethod
    def squareUniform(params: SampleGenerationParameters) -> np.ndarray:
        """ Single Cosine Wave - Params from Uniform Distrobution """
        x = np.zeros(shape=(params.inputAxis.size,),dtype=np.float32)
        waveCount = np.random.uniform(low=params.waveCountLow,low=params.waveCountHigh,size=1)
        a = np.random.Generator.uniform(params.amplitudeLow,params.amplitudeHigh,size=waveCount)
        f = np.random.Generator.uniform(params.frequencyLow,params.frequencyHigh,size=waveCount)
        p = np.random.Generator.uniform(params.phaseLow,params.phaseHigh,size=waveCount)
        o = np.random.Generator.uniform(params.offsetLow,params.offsetHigh,size=waveCount)
        for ii in range(waveCount):
            x += a[ii] * scisig.square(t=(2 * np.pi * params.inputAxis * f[ii]) + p[ii],duty=0.5) + o[ii] 
        return x

    @staticmethod
    def sawtoothUniform(params: SampleGenerationParameters) -> np.ndarray:
        """ Single Cosine Wave - Params from Uniform Distrobution """
        x = np.zeros(shape=(params.inputAxis.size,),dtype=np.float32)
        waveCount = np.random.uniform(low=params.waveCountLow,low=params.waveCountHigh,size=1)
        a = np.random.Generator.uniform(params.amplitudeLow,params.amplitudeHigh,size=waveCount)
        f = np.random.Generator.uniform(params.frequencyLow,params.frequencyHigh,size=waveCount)
        p = np.random.Generator.uniform(params.phaseLow,params.phaseHigh,size=waveCount)
        o = np.random.Generator.uniform(params.offsetLow,params.offsetHigh,size=waveCount)
        for ii in range(waveCount):
            x += a[ii] * scisig.sawtooth(t=(2 * np.pi * params.inputAxis * f[ii]) + p[ii],width=1) + o[ii] 
        return x

    @staticmethod
    def squareUniform(params: SampleGenerationParameters) -> np.ndarray:
        """ Single Cosine Wave - Params from Uniform Distrobution """
        x = np.zeros(shape=(params.inputAxis.size,),dtype=np.float32)
        waveCount = np.random.uniform(low=params.waveCountLow,low=params.waveCountHigh,size=1)
        a = np.random.Generator.uniform(params.amplitudeLow,params.amplitudeHigh,size=waveCount)
        f = np.random.Generator.uniform(params.frequencyLow,params.frequencyHigh,size=waveCount)
        p = np.random.Generator.uniform(params.phaseLow,params.phaseHigh,size=waveCount)
        o = np.random.Generator.uniform(params.offsetLow,params.offsetHigh,size=waveCount)
        for ii in range(waveCount):
             x += a[ii] * scisig.sawtooth(t=(2 * np.pi * params.inputAxis * f[ii]) + p[ii],width=0.5) + o[ii] 
        return x

class SampleGenerator:
    """ 
        Base Class for All Sample Generators
    """

    def __init__(self,
                 className: str,
                 classIndex: int,
                 drawLimit: int,
                 sampleGeneratorCallback: function,
                 waveformParams: SampleGenerationParameters,):
        """ Constructor """
        self._name      = className
        self._index     = classIndex
        self._drawCount = 0
        self._capacity  = drawLimit

        self._generatorCallback  = sampleGeneratorCallback
        self._waveformParams     = waveformParams

    def __del__(self):
        """ Destructor """
        pass

    # Accessors
    
    def drawCount(self) -> int:
        """ Return the number of  draws """
        return self._drawCount

    def drawLimit(self) -> int:
        """ Return the limit on the number of draws """
        return self._capacity

    def isEmpty(self) -> bool:
        """ Return if the drawLimit has been reached """
        return (self._drawCount >= self._capacity)

    def params(self) -> SampleGenerationParameters:
        """ Return sample generation parameters """
        return self._waveformParams

    # Public Interface

    def draw(self) -> np.ndarray:
        """ Draw a sample """
        if (self.isEmpty() == True):
            msg = "Draw limit reached on {0}".format(self)
            raise RuntimeError(msg)
        sample = self.__generateSample()
        return sample

    def resetDrawCount(self) -> None:
        """ Reset the internal draw counter """
        self._drawCount = 0
        return None

    # Private

    def __generateSample(self) -> np.ndarray:
        """ Invoke the callback to generate a sample """
        return self._generatorCallback.__call__(self._waveformParams)

    # Dunder

    def __str__(self) -> str:
        """ Cast to string """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0}: ({1}/{2})".format(str(self),self._drawCount,self._capacity)


