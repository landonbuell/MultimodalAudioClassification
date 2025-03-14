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

def __generateUniformComponentWaves(
        params: sampleGeneratorTypes.SampleGenerationParameter) -> list:
    """ Generate uniform parameters, return as a list """
    numWaves = np.random.randint(low=params.waveCount.low,high=params.waveCount.high)
    components = [sampleGeneratorTypes.GeneratedSample.ComponentWaveConfig() for x in range(numWaves)]
    generator = np.random.default_rng()
    # Populate the list of component waves
    for ii in range(len(components)):
        components[ii].amp      = generator.uniform(low=params.amp.low,high=params.amp.high)
        components[ii].freq     = generator.uniform(low=params.freq.low,high=params.freq.high)
        components[ii].phase    = generator.uniform(low=params.phase.low,high=params.phase.high)
        components[ii].off      = generator.uniform(low=params.off.low,high=params.off.high)
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

