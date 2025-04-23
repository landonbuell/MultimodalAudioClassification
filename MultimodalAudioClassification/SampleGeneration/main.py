"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    SampleGeneration
File:       WaveformGenerators.py

Author:     Landon Buell
Date:       January 2023
"""


        #### IMPORTS ####

import sys

import numpy as np
import scipy as sp

import sampleGenerator
import sampleGeneratorCallbacks
import sampleGeneratorTypes


        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set Some Params
    np.random.seed( 123456789 )
    generationParams = sampleGeneratorTypes.SampleGenerationParameters()
    
    """
    # Cosine sample generator
    cosineUniform = sampleGeneratorCallbacks.Uniform(np.cos)
    cosineUniformGenerator = sampleGenerator.SampleGenerator(
        params=generationParams,
        callback=cosineUniform)

    while (cosineUniformGenerator.isEmpty() == False):
        generatedSample = cosineUniformGenerator.drawNext()
        3print(str(generatedSample))
        generatedSample.showWaveform()
    """

    # Square wave sample generator
    squareUniform = sampleGeneratorCallbacks.Uniform(sp.signal.square)
    squareUniformGenerator = sampleGenerator.SampleGenerator(
        params=generationParams,
        callback=squareUniform)

    while (squareUniformGenerator.isEmpty() == False):
        generatedSample = squareUniformGenerator.drawNext()
        print(str(generatedSample))
        generatedSample.showWaveform()

    sys.exit(0)




