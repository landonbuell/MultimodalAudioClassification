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

def __generateUniformParams(params: sampleGeneratorTypes.SampleGenerationParameter) -> list:
    """ Generate uniform parameters, return as a list """
    numWaves = np.random.randint(low=params.waveCount.low,high=params.waveCount.high)
    waveParams = [sampleGeneratorTypes.GeneratedSample.Parameters() for x in range(numWaves)]
    generator = np.random.default_rng()



    return None

def cosineUniform(params: sampleGeneratorTypes.SampleGenerationParameters) -> sampleGeneratorTypes.GeneratedSample:
    """ Cosine Wave(s) - Params from Uniform Distrobution """
    return None

def squareUniform(params: sampleGeneratorTypes.SampleGenerationParameters) -> sampleGeneratorTypes.GeneratedSample:
    """ Single Cosine Wave - Params from Uniform Distrobution """
    return None

def sawtoothUniform(params: sampleGeneratorTypes.SampleGenerationParameters) -> sampleGeneratorTypes.GeneratedSample:
    """ Single Cosine Wave - Params from Uniform Distrobution """
    return None

def squareUniform(params: sampleGeneratorTypes.SampleGenerationParameters) -> sampleGeneratorTypes.GeneratedSample:
    """ Single Cosine Wave - Params from Uniform Distrobution """
    return None

