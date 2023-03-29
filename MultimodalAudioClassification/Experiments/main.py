"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    Experiments
File:       main.py

Author:     Landon Buell
Date:       November 2022
"""

    #### IMPORTS ####

import sys
import os

import numpy as np

import KFoldsCrossValidation

import PyToolsStructures
import Preprocessors
import Experiments



    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set some constants + Load Run Info
    FEATURES = "C:\\Users\\lando\\Documents\\audioFeatures\\allSamplesV3"
    OUTPUT = "C:\\Users\\lando\\Documents\\audioPredictions\\allSamplesV3"

    runInfo = PyToolsStructures.RunInformation.deserialize(FEATURES)
    NUM_CLASSES = 32

    # Multimodal Nueral Network Experiment


