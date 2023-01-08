"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    Experiments
File:       ExperimentDrivers.py

Author:     Landon Buell
Date:       November 2022
"""

    #### IMPORTS ####

import sys
import os

import numpy as np

import PyToolsStructures

    #### CONSTANTS ####


    #### CLASS DEFINITIONS ####

class BaseExperiment:
    """ Parent Class for Running Experiments """

    def __init__(self,
                 runInfo,
                 outputPath,
                 trainSize=0.8,
                 numIters=1,
                 seed=0,
                 batchesAtOnce=64,
                 pipelinesToUse=None):
        """ Constructor """
        self._runInfo       = runInfo
        self._outputPath    = os.path.abspath(outputPath)
        
        self._numIters      = numIters
        self._randomSeed    = seed
        
        self._batchesAtOnce     = batchesAtOnce
        self._trainingBatches   = []
        self._testingBatches    = []

        self._pipelines     = [False] * PyToolsStructures.RunInformation.DEFAULT_NUM_PIPELINES;
        self._model         = None          

        for pipelineIndex in pipelinesToUse:
            self._pipelines[pipelineIndex] = True
        np.random.seed(self._randomSeed)
        
    def __del__(self):
        """ Destructor """
        self._runInfo = None
        
    # Getters and Setters

    # Public Interface

    def run(self):
        """ Run the Experiment """

        for ii in range(self._numIters):

            # Initialize Model + Train Test Split
            self.__initializeModel()
            self.__executeTrainTestSplit()

            # Load + Train the Model
            self.__runLoadAndTrainSequence()

            # Load + Predict on the Model
            self.__runLoadAndTestSequence()

            # Export Experiment Details
            self.__exportExperimentDetails(ii)

        return self

    def resetState(self):
        """ Public accessor to reset the state of the experiment instance """
        return self.__resetState()

    def predictWithModel(self,X):
        """ Run predictions on Model """

        return self

    # Protected Interface

    def _generateModel(self):
        """ ABSTRACT - Generate a model to run the experiment with """
        return None

    def _loadSamplesFrom(self):
        """ VIRTUAL - Load Data for training """
        batchesToLoad = []
        for ii in range(self._batchesAtOnce):
            batchIndex = self._batchesRemaining.pop()
            batchesToLoad.append(batchIndex)

        return (X,y)

    def _trainModel(self,X,y):
        """ VIRTUAL - Train the model with the provided data """


        return self
    
    def _testModel(self,X,y):
        """ VIRTUAL - Test the model on the provided data """

        return self

    # Private Interface

    def __resetState(self):
        """ Reset the state of the experiment """
        self._randomSeed += 1
        np.random.seed(self._randomSeed)

        self._trainingBatches   = []
        self._testingBatches    = []

        return self
    
    def __initializeModel(self):
        """ Initialize the Neural Network Model """
        return self

    def __executeTrainTestSplit(self):
        """ Perform train-test split on batches """
        allBatches = np.random.permutation(
            self._runInfo.getNumBatches())
        # split
        numTrainBatches = np.floor(len(allBatches))
        numTestBatches = len(allBatches) - numTrainBatches
        # Add to batches
        for ii in range(numTrainBatches):
            self._trainingBatches.append( allBatches.pop() )
        self._testingBatches = allBatches[:]
        return self

    def __runLoadAndTrainSequence(self):
        """ Run data loading/training sequence """
        return self

    def __runLoadAndTestSequence(self):
        """ Run data loading/testing sequence """
        return self

    def __exportExperimentDetails(self,iterCounter):
        """ Export the Details of the experient """
        return self

