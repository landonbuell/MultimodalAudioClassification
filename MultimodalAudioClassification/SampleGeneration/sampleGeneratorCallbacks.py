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
    """ Base callback for Sample Generator """

    def __init__(self,
                 callback: function,
                 params: sampleGeneratorTypes.SampleGenerationParameters):
        """ Constructor """
        self._callable  = None
        self._params    = params

    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    def __call__(self):
        """ Override call operator """
        return None

    # Protected Methods

    def __generateUniformComponentWaves() -> list:
        """ Generate uniform parameters, return as a list """
        numWaves = np.random.randint(low=self._params.waveCount.low,high=self._params.waveCount.high)
        components = [sampleGeneratorTypes.GeneratedSample.ComponentWaveConfig() for x in range(numWaves)]
        generator = np.random.default_rng()
        # Populate the list of component waves
        for ii in range(len(components)):
            components[ii].amp      = generator.uniform(low=self._params.amp.low,high=self._params.amp.high)
            components[ii].freq     = generator.uniform(low=self._params.freq.low,high=self._params.freq.high)
            components[ii].phase    = generator.uniform(low=self._params.phase.low,high=self._params.phase.high)
            components[ii].off      = generator.uniform(low=self._params.off.low,high=self._params.off.high)
        return components

    def __assembleWaveformFromComponents(
            listOfComponentWaveConfigs: list,
            inputAxis: np.ndarray) -> np.ndarray:
        """ Use provided wave configs to assemble a waveform """
        waveform = np.zeros(inputAxis.size)
        for ii,config in enumerate(listOfComponentWaveConfigs):
            a,f,p,o = config.unpack()
            waveform += a * config.func(2*np.pi*f*inputAxis - p) + o
        sample = sampleGeneratorTypes.GeneratedSample(
            waveform, params=listOfComponentWaveConfigs)
        return sample

class CosineUniform(SampleGenerationCallback):
    """
        Generate Composite Cosone Waves w/ Uniformly distributed parameters
    """

    def __init__(self,
                 params: sampleGeneratorTypes.SampleGenerationParameters):
        """ Constructor """
        super().__init__(np.cos,params)

    def __del__(self):
        """ Destructor """
        super().__del__()

    def 
        

def cosineUniform(params: sampleGeneratorTypes.SampleGenerationParameters) -> sampleGeneratorTypes.GeneratedSample:
    """ Cosine Wave(s) - Params from Uniform Distrobution """
    componentWaveParams = __generateUniformComponentWaves(params)
    for ii in range(len(componentWaveParams)):
        componentWaveParams[ii].func = np.cos
    sample = __assembleWaveformFromComponents(componentWaveParams,params.inputAxis)
    return sample

def squareUniform(params: sampleGeneratorTypes.SampleGenerationParameters) -> sampleGeneratorTypes.GeneratedSample:
    """ Single Cosine Wave - Params from Uniform Distrobution """
    return None

def sawtoothUniform(params: sampleGeneratorTypes.SampleGenerationParameters) -> sampleGeneratorTypes.GeneratedSample:
    """ Single Cosine Wave - Params from Uniform Distrobution """
    return None

def squareUniform(params: sampleGeneratorTypes.SampleGenerationParameters) -> sampleGeneratorTypes.GeneratedSample:
    """ Single Cosine Wave - Params from Uniform Distrobution """
    return None

