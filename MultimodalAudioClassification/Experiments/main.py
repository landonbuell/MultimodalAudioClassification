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
    FEATURES = "C:\\Users\\lando\\Documents\\audioFeatures\\allSamplesV2"
    OUTPUT = "C:\\Users\\lando\\Documents\\audioPredictions\\allSamplesV2"

    runInfo = PyToolsStructures.RunInformation.deserialize(FEATURES)
    NUM_CLASSES = 4

    # Multilayer Perceptron Experiment
    mlpExp = Experiments.MultilayerPerceptronExperiment(
        runInfo,os.path.join(OUTPUT,"mlp"))
    #mlpExp.run()
    

    # Convolutional Neural Network Experiment
    cnnExp = Experiments.ConvolutionalNeuralNetworkExperiment(
        runInfo,os.path.join(OUTPUT,"cnn"))
    cnnExp.run()

    # Multimodal Nueral Network Experiment


