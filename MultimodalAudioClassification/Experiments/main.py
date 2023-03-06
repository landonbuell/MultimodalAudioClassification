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


import NeuralNetworks




    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set some constants + Load Run Info
    FEATURES = "C:\\Users\\lando\\Documents\\audioFeatures\\allsamplesV1"
    OUTPUT = "C:\\Users\\lando\\Documents\\audioPredictions\\allsamplesV1"

    runInfo = PyToolsStructures.RunInformation.deserialize(FEATURES)
    NUM_CLASSES = 4

    # Multilayer Perceptron Experiment
    

    # Convolutional Neural Network Experiment

    # Multimodal Nueral Network Experiment

    xValidation = KFoldsCrossValidation.KFoldsCrossValidation(
        runInfo,
        OUTPUT,
        numFolds=10,
        modelLoaderCallback=None,
        dataLoaderCallback=None,
        seed=987654321)
    xValidation.run()

