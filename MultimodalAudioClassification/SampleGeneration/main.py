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
import sampleGeneratorCallbacks
import sampleGeneratorTypes


        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set Some Params
    np.random.seed( 123456789 )
    generationParams = sampleGeneratorTypes.SampleGenerationParameters()
    cosineUniform = sampleGeneratorCallbacks.Uniform(np.cos)

    cosineUniformGenerator = sampleGenerator.SampleGenerator(
        params=generationParams,
        callback=cosineUniform)

    while (cosineUniformGenerator.isEmpty() == False):
        generatedSample = cosineUniformGenerator.drawNext()
        print(str(generatedSample))

    sys.exit(0)




