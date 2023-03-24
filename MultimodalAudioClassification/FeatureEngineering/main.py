"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    FeautureEngineering
File:       main.py

Author:     Landon Buell
Date:       Sept 2022
"""

    #### IMPORTS ####

import sys
import os

import PyToolsStructures
import Preprocessors

    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set some constants + Load Run Info
    INPUT_PATH  = "C:\\Users\\lando\\Documents\\audioFeatures\\allSamplesV2"
    runInfo = PyToolsStructures.RunInformation.deserialize(INPUT_PATH)

    # Create + Fit the Scaler Instance
    #scaler = Preprocessors.CustomStandardScaler(runInfo)
    #scaler.fit()

    # Create + Fit the Scaler Wrapper Instance 
    scalerWrapper = Preprocessors.StandardScalerWrapper(runInfo)
    scalerWrapper.fit()

    sys.exit(0)