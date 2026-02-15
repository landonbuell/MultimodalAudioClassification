"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    SampleGeneration
    File:       sampleGeneratorPresets.py
    Classes:    SampleGenerator

    Author:     Landon Buell
    Date:       June 2025
"""

    #### IMPORTS ####

import numpy as np
import scipy as sp

import sampleGenerator
import sampleGeneratorCallbacks
import sampleGeneratorTypes

    #### CLASS DEFINITIONS ####

def getUniformSquare(sampleLimit: int, classIndex: int):
    """ Get square wave with component waves drawn from uniform distribution """
    generationParams = sampleGeneratorTypes.SampleGenerationParameters()
    squareUniform = sampleGeneratorCallbacks.Uniform(sp.signal.square)
    squareUniformGenerator = sampleGenerator.SampleGenerator(
        params=generationParams,
        callback=squareUniform,
        drawLimit=sampleLimit,
        classIndex=classIndex,
        className="UniformSquare")
    return squareUniformGenerator

def getUniformSawtooth(sampleLimit: int, classIndex: int):
    """ Get sawtooth wave with component waves drawn from uniform distribution """
    generationParams = sampleGeneratorTypes.SampleGenerationParameters()
    squareUniform = sampleGeneratorCallbacks.Uniform(sp.signal.sawtooth)
    squareUniformGenerator = sampleGenerator.SampleGenerator(
        params=generationParams,
        callback=squareUniform,
        drawLimit=sampleLimit,
        classIndex=classIndex,
        className="UniformSawtooth")
    return squareUniformGenerator

def getUniformSine(sampleLimit: int, classIndex: int):
    """ Get sine wave with component waves drawn from uniform distribution """
    generationParams = sampleGeneratorTypes.SampleGenerationParameters()
    squareUniform = sampleGeneratorCallbacks.Uniform(np.sin)
    squareUniformGenerator = sampleGenerator.SampleGenerator(
        params=generationParams,
        callback=squareUniform,
        drawLimit=sampleLimit,
        classIndex=classIndex,
        className="UniformSine")
    return squareUniformGenerator

def getUniformCosine(sampleLimit: int, classIndex: int):
    """ Get cosine wave with component waves drawn from uniform distribution """
    generationParams = sampleGeneratorTypes.SampleGenerationParameters()
    squareUniform = sampleGeneratorCallbacks.Uniform(np.cos)
    squareUniformGenerator = sampleGenerator.SampleGenerator(
        params=generationParams,
        callback=squareUniform,
        drawLimit=sampleLimit,
        classIndex=classIndex,
        className="UniformCosine")
    return squareUniformGenerator