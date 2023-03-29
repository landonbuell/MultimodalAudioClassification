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
import KFoldsCrossValidation

    #### FUNCTION DEFINITIONS ####

def runSingleMultilayerPercepton(runInfo,outputPath,seed=0):
    """ Run + Export single instance of multilayer perceptron experiment """
    experiment = Experiments.MultilayerPerceptronExperiment(runInfo,outputPath,seed=seed)
    experiment.run()
    return None

def runSingleConvolutionalNeuralNetworkExperiment(runInfo,outputPath,seed=0):
    """ Run + Export single instance of convolutional neural network experiment """
    experiment = Experiments.ConvolutionalNeuralNetworkExperiment(runInfo,outputPath,seed=seed)
    experiment.run()
    return None

def runSingleHybridNeuralNetworkExperiment(runInfo,outputPath,seed=0):
    """ Run + Export single instace of Hybrid Neural Network experiment """
    experiment = Experiments.HybridNeuralNetworkExperiment(runInfo,outputPath,seed=seed)
    experiment.run()
    return None

def runKfoldsMultilayerPerceptron(runInfo,outputPath,seed=0,folds=10):
    """ Run + Export K-Folds on Multilayer Pereptron """
    experiment = Experiments.MultilayerPerceptronExperiment(runInfo,outputPath,seed=seed)
    xFoldsValidator = KFoldsCrossValidation.KFoldsCrossValidation(experiment,10)
    xFoldsValidator.run()
    return None


    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set some constants + Load Run Info
    FEATURES = "C:\\Users\\lando\\Documents\\audioFeatures\\allSamplesV3"
    OUTPUT = "C:\\Users\\lando\\Documents\\audioPredictions\\allSamplesV3"

    runInfo = PyToolsStructures.RunInformation.deserialize(FEATURES)

    # K-Folds on Multilayer Perceptron
    runKfoldsMultilayerPerceptron(runInfo,OUTPUT)

    # K-Folds on Convolutional Neural Network

    # K-Folds on Hybrid Neural Network


