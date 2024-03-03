"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    SampleGeneration
    File:       simpleWaveforms.py
    Classes:    __SimpleWaveform,
                SineWave,
                SquareWave,
                TriangleWave,
                SawtoothWave

    Author:     Landon Buell
    Date:       March 2024
"""

        #### IMPORTS ####

import numpy as np

        #### CLASS DEFINITIONS ####

class SimpleWaveformGeneratorParams:
    """ Structure to store params for a simple waveform generator """

    def __init__(self):
        """ Constructor """
        self.amplitudeLow   = 1.0
        self.amplitudeHigh  = 10.0
        self.frequencyLow   = 1.0
        self.frequencyHigh  = 10.0
        self.phaseLow       = 0.0
        self.phaseHigh      = 2.0*np.pi
        self.offsetLow      = -1.0
        self.offsetHigh     = 1.0

class __SimpleWaveform:
    """
        Abstract base class for simple waveform generators 
    """

    def __init__(self,
                    waveformCallback,   # Generates waveform
                    waveformParams: SimpleWaveformGeneratorParams):
        """ Constructor """
        self._waveformCallback  = waveformCallback
        self._params            = waveformParams       

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getParams(self) -> SimpleWaveformGeneratorParams:
        """ Return the underlying parameters structure """
        return self._params

    # Public Interface

    def draw(self,
             numSamples: int) -> np.ndarray:
        """ Draw a random sample for each waveform parameter and return a waveform """
        x = np.arange(0,numSamples) / 0.01
        a = np.random.Generator.uniform(self._params.amplitudeLow,self._params.amplitudeHigh,size=1)
        f = np.random.Generator.uniform(self._params.frequencyLow,self._params.frequencyHigh,size=1)
        p = np.random.Generator.uniform(self._params.phaseLow,self._params.phaseHigh,size=1)
        o = np.random.Generator.uniform(self._params.offsetLow,self._params.offsetHigh,size=1)
        # TODO: Finish this!
        return x
