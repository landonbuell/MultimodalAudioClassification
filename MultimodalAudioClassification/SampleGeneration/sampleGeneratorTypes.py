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
        s = "{0} @ {1} w/ {2} waves foe class #{3}".format(
            self.__class__,
            hex(id(self)),
            self.waveCount,
            self.classInt)
        return s

class SampleGeneratorCallbacks:
    """ 
        Static class of callbacks to generate samples 
        All signatures must obey:

        def callbackName(
            params: SampleGenerationParameters) -> GeneratedSample
       
    """

    @staticmethod
    def cosineUniform(params: SampleGenerationParameters) -> GeneratedSample:
        """ Cosine Wave(s) - Params from Uniform Distrobution """
        x = np.zeros(shape=(params.inputAxis.size,),dtype=np.float32)
        waveCount = np.random.uniform(low=params.waveCountLow,low=params.waveCountHigh,size=1)
        params = [GeneratedSample.Params() for x in range(waveCount)]
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
    def squareUniform(params: SampleGenerationParameters) -> GeneratedSample:
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
    def sawtoothUniform(params: SampleGenerationParameters) -> GeneratedSample:
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
    def squareUniform(params: SampleGenerationParameters) -> GeneratedSample:
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