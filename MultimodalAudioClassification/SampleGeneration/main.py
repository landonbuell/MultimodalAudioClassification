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

import sampleGenerator
import sampleGeneratorTypes
import generatorCallbacks



        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set Some Params
    np.random.seed( 123456789 )
    generationConfig    = sampleGeneratorTypes.SampleGeneratorConfig(
        "COSINE",1,1024,generatorCallbacks.cosineUniform)
    generationParams = sampleGeneratorTypes.SampleGenerationParameters()

    cosineGenerator = sampleGenerator.SampleGenerator(
        config=generationConfig,
        params=generationParams)

    while (cosineGenerator.isEmpty() == False):
        generatedSample = cosineGenerator.drawNext()
        print(str(generatedSample))

    sys.exit(0)




