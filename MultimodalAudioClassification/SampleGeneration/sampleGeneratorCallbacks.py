"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    SampleGeneration
    File:       generatorCallbacks.py
    Classes:    NONE

    Author:     Landon Buell
    Date:       February 2025
"""

        #### IMPORTS ####

import numpy as np

import sampleGeneratorTypes

        #### FUNCTION DEFINITIONS ####

class SampleGenerationCallback:
    """ Base callback for Sample Generator. Abstract - Make no Instance """

    def __init__(self,
                 callback,
                 seed=123456789):
        """ Constructor """
        self._callable  = None
        self._generator = np.random.default_rng(seed=seed)

    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    def __call__(self,
                 params: sampleGeneratorTypes.SampleGenerationParameters) -> sampleGeneratorTypes.GeneratedSample:
        """ Virtual:  call operator """

        waveform = self.__assembleWaveformFromComponents(listOfComponentWaveParams,
                                                         params.inputAxis)
        sample = sampleGeneratorTypes.GeneratedSample(waveform,-1,listOfComponentWaveParams)
        return sample

    # Private Methods

    def __generateUniformComponentWaves(self,
                                       params: sampleGeneratorTypes.SampleGenerationParameters ) -> list:
        """ Generate uniform parameters, return as a list """
        numWaves = np.random.randint(low=self._params.waveCount.low,high=self._params.waveCount.high)
        components = [sampleGeneratorTypes.GeneratedSample.ComponentWaveConfig() for x in range(numWaves)]
        amps    = self._generator.uniform(low=self._params.amp.low,high=self._params.amp.high,size=numWaves)
        freqs   = self._generator.uniform(low=self._params.freq.low,high=self._params.freq.high,size=numWaves)
        phases  = self._generator.uniform(low=self._params.phase.low,high=self._params.phase.high,size=numWaves)
        offsets = self._generator.uniform(low=self._params.off.low,high=self._params.off.high,size=numWaves)
        # Attach to structures
        for ii in range(numWaves):
            components[ii].amp      = amps[ii]
            components[ii].freq     = freqs[ii]
            components[ii].phase    = phases[ii]
            components[ii].off      = offsets[ii]
        return components

    def __assembleWaveformFromComponents(self,
            listOfComponentWaveConfigs: list,
            inputAxis: np.ndarray) -> np.ndarray:
        """ Use provided wave configs to assemble a waveform """
        waveform = np.zeros(inputAxis.size)
        for ii,config in enumerate(listOfComponentWaveConfigs):
            a,f,p,o = config.unpack()
            waveform += a * self._callable(2*np.pi*f*inputAxis - p) + o
        return waveform

class Uniform(SampleGenerationCallback):
    """
        Generate Composite Waves w/ Uniformly distributed parameters
    """

    def __init__(self,
                 params: sampleGeneratorTypes.SampleGenerationParameters):
        """ Constructor """
        super().__init__(np.cos,params)

    def __del__(self):
        """ Destructor """
        super().__del__()


        
