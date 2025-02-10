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



        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set Some Params
    np.random.seed( 123456789 )
    generationParams = sampleGeneratorTypes.SampleGenerationParameters()

    cosineGenerator = sampleGenerator.SampleGenerator(
        className="COSINE",
        classIndex=1,
        drawLimit=1024,
        sampleGeneratorCallback=sampleGeneratorTypes.SampleGeneratorCallbacks.cosineUniform,
        waveformParams=generationParams)

    while (cosineGenerator.isEmpty() == False):
        generatedSample = cosineGenerator.drawNext()
        print(str(generatedSample))

    sys.exit(0)




