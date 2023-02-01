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

class __BaseExperiment:
    """ Parent Class for Running Experiments """

    def __init__(self,
                 runInfo,
                 outputPath,
                 model,
                 dataloaderCallback,
                 numIters=1,
                 seed=123456789):
        """ Constructor """
        self._runInfo       = runInfo
        self._outputPath    = os.path.abspath(outputPath)
        
        self._model         = model
        self._dataCallback  = dataloaderCallback

        self._numIters      = numIters
        self._seed          = seed

        self._trainingBatches   = np.array([],dtype=np.int32)
        self._testingBatches    = np.array([],dtype=np.int32)
        
    def __del__(self):
        """ Destructor """
        self._runInfo   = None
        self._model     = None

        
    # Getters and Setters

    # Public Interface

    def registerTrainingBatches(self,batches):
        """ Add a list of batches to the training batch list """
        self._trainingBatches = np.append(self._trainingBatches,batches)
        return self

    def registerTestingBatches(self,batches):
        """ Add a list of batches to the testing batch list """
        self._testingBatches = np.append(self._testingBatches,batches)
        return self


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
        self._model = self._config.initializeModel()
        return self

    def __executeTrainTestSplit(self):
        """ Perform train-test split on batches """
        self._trainingBatches   = self._config.sendTrainBatches(self._runInfo)
        self._testingBatches    = self._config.sendTestBatches(self._runInfo)
        return self

    def __runLoadAndTrainSequence(self):
        """ Run data loading/training sequence """
        for batch in self._trainingBatches:
            designMatrices = None

        return self

    def __runLoadAndTestSequence(self):
        """ Run data loading/testing sequence """
        return self

    def __exportExperimentDetails(self,iterCounter):
        """ Export the Details of the experient """
        return self


class CrossValidationFoldExperiment(__BaseExperiment):
    """ Experiment to be used as a apart of a X-Validation set """

    def __init__(self,
                 runInfo,
                 outputPath,
                 model,
                 dataloaderCallback,
                 foldIndex,
                 numIters=1,
                 seed=123456789):
        """ Constructor """
        super().__init__(runInfo,
                         outputPath,
                         model,
                         dataloaderCallback,
                         numIters,
                         seed)
        self._foldIndex = foldIndex
        
    def __del__(self):
        """ Destructor """
        super().__del__()

