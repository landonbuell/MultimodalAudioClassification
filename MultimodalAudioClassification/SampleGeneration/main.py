"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    SampleGeneration
File:       WaveformGenerators.py

Author:     Landon Buell
Date:       January 2023
"""


        #### IMPORTS ####

import os
import sys

import numpy as np
import WaveformGenerators

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set Some Params
    np.random.seed( 123456789 )
    OUTPUT_PATH = "C:\\Users\\lando\\Documents\\audioSyntheicBinaries"
    NUM_AUDIO_SAMPLES = 1000
    SAMPLE_RATE = 44100
    TIME = 4
    timeAxis = np.arange(0,2,1/SAMPLE_RATE,dtype=np.float32)
    
    # Generate Sine Waves
    sineWaveGenerator = WaveformGenerators.DatasetGenerator(
        WaveformGenerators.SimpleWavesforms.getSineWave,
        WaveformGenerators.SimpleNoise.getUniformNoise,
        timeAxis,
        name="sineWave")
    sineWaveGenerator.createSamples(NUM_AUDIO_SAMPLES,OUTPUT_PATH)





