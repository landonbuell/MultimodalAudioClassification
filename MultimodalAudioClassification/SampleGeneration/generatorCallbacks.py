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

def __generateUniformComponentWaves(params: sampleGeneratorTypes.SampleGenerationParameter) -> list:
    """ Generate uniform parameters, return as a list """
    numWaves = np.random.randint(low=params.waveCount.low,high=params.waveCount.high)
    components = [sampleGeneratorTypes.GeneratedSample.ComponentWave() for x in range(numWaves)]
    generator = np.random.default_rng()
    # Populate the list of component waves
    for ii in range(len(components)):
        components[ii].amp = generator.uniform(low=params.amp.low,high=params.amp.high)
        components[ii].freq = generator.uniform(low=params.freq.low,high=params.freq.high)
        components[ii].phase = generator.uniform(low=params.phase.low,high=params.phase.high)
        components[ii].off = generator.uniform(low=params.off.low,high=params.off.high)
    return components

def cosineUniform(params: sampleGeneratorTypes.SampleGenerationParameters) -> sampleGeneratorTypes.GeneratedSample:
    """ Cosine Wave(s) - Params from Uniform Distrobution """
    componentWaveParams = __generateUniformComponentWaves(params)
    waveform = np.zeros(params.inputAxis.size)
    # Aggregate waves
    for ii,wave in enumerate(componentWaveParams):
        a,f,p,o = wave.unpack()
        waveform += a * np.cos(2*np.pi*f*params.inputAxis - p) + o
    sample = sampleGeneratorTypes.GeneratedSample(
        waveform, params=componentWaveParams)
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

