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
        self._callable  = callback
        self._generator = np.random.default_rng(seed=seed)

    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    def __call__(self,
                 params: sampleGeneratorTypes.SampleGenerationParameters) -> sampleGeneratorTypes.GeneratedSample:
        """ Virtual:  call operator """
        listOfComponentWaveParams = self._generateComponentWaves(params)
        waveform = self.__assembleWaveformFromComponents(listOfComponentWaveParams,
                                                         params.inputAxis)
        sample = sampleGeneratorTypes.GeneratedSample(waveform,-1,listOfComponentWaveParams)
        return sample

    # Protected Methods

    def _generateComponentWaves(self,
                                params: sampleGeneratorTypes.SampleGenerationParameters ) -> list:
        """ VIRTUAL: Generate uniform parameters, return as a list """
        numWaves = np.random.randint(low=params.waveCount.low,high=params.waveCount.high)
        components = [sampleGeneratorTypes.GeneratedSample.ComponentWaveConfig() for x in range(numWaves)]
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
                 callback,
                 seed=123456789):
        """ Constructor """
        super().__init__(callback,seed)

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Protected Methods

    def _generateComponentWaves(self,
                                params: sampleGeneratorTypes.SampleGenerationParameters ) -> list:
        """ OVERRIDE: Generate uniform parameters, return as a list """
        components = super()._generateComponentWaves(params)
        numWaves = len(components)
        # Generate Uniform parameters
        amps    = self._generator.uniform(low=params.amp.low,high=params.amp.high,size=numWaves)
        freqs   = self._generator.uniform(low=params.freq.low,high=params.freq.high,size=numWaves)
        phases  = self._generator.uniform(low=params.phase.low,high=params.phase.high,size=numWaves)
        offsets = self._generator.uniform(low=params.off.low,high=params.off.high,size=numWaves)
        # Attach to structures
        for ii in range(numWaves):
            components[ii].amp      = amps[ii]
            components[ii].freq     = freqs[ii]
            components[ii].phase    = phases[ii]
            components[ii].off      = offsets[ii]
        return components
        
