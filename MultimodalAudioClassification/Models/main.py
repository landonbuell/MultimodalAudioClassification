"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    Models
File:       main.py

Author:     Landon Buell
Date:       November 2022
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
    FEATURES = "C:\\Users\\lando\\Documents\\audioFeatures\\simpleSignalsV1"
    PROCESSED = os.path.join(FEATURES,"preprocessedV1")
    runInfo = PyToolsStructures.RunInfo.deserialize(FEATURES)

    # Multilayer Perceptron Experiment

    # Convolutional Neural Network Experiment

    # Multimodal Nueral Network Experiment


