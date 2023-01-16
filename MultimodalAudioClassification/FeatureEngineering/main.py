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

import numpy as np


    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set some constants + Load Run Info
    INPUT_PATH  = "C:\\Users\\lando\\Documents\\audioFeatures\\simpleSignalsV1"
    OUTPUT_PATH = "C:\\Users\\lando\\Documents\\audioFeatures\\simpleSignalsV1_processed"
    runInfo = PyToolsStructures.RunInformation.deserialize(INPUT_PATH)

    # Create + Fit the Scaler Instance
    scaler = Preprocessors.StandardScaler(runInfo,OUTPUT_PATH)
    scaler.fit()


    sys.exit(0)