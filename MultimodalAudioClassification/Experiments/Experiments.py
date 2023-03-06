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

import ExperimentCallbacks
import ModelParams

import PyToolsStructures


    #### CONSTANTS ####


    #### CLASS DEFINITIONS ####

class __BaseExperiment:
    """ Parent Class for Running Experiments """

    def __init__(self,
                 runInfo,
                 outputPath,
                 modelLoaderCallback,   # <model> = modelLoaderCallback.__call__(self,randomSeed)
                 dataloaderCallback,    # <(X,y)> = dataloaderCallback.__call__(self,batchIndex)
                 pipelines,
                 trainSize=0.8,
                 numIters=1,              
                 seed=123456789):
        """ Constructor """
        self._runInfo       = runInfo
        self._outputPath    = os.path.abspath(outputPath)
        
        self._modelLoaderCallback   = modelLoaderCallback
        self._dataLoaderCallback    = dataloaderCallback
        
        self._numIters      = numIters
        self._seed          = seed

        self._pipelines     = pipelines
        self._trainSize     = trainSize

        self._model     = None
        self._fitParams = ModelParams.TensorFlowFitModelParams()

        self._trainingBatches   = np.array([],dtype=np.int32)
        self._testingBatches    = np.array([],dtype=np.int32)

        self._trainingHistories = []
        
    def __del__(self):
        """ Destructor """
        self._runInfo   = None
        self._model     = None

        
    # Getters and Setters

    def getRunInfo(self):
        """ Return the RunInfo Structure """
        return self._runInfo

    def getPipelines(self):
        """ Return a list of the pipelines to load """
        return self._pipelines

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
            self.__registerTrainTestBatches()

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
    
    def __initializeModel(self):
        """ Initialize the Neural Network Model """
        randomState = self._seed
        self._model = self._modelLoaderCallback.__call__(randomState)
        return self

    def __loadBatch(self,batchIndex):
        """ Load + Return a Batch of Data """
        X,Y = self._dataLoaderCallback.__call__(self,batchIndex)
        return (X,Y)

    def __registerTrainTestBatches(self):
        """ Determine which batches will be used for training/testing """
        totalNumBatches = self._runInfo.getNumBatches()
        numTrainBatches = int(totalNumBatches * self._trainSize)

        batchIndicies = np.random.shuffle(np.arange(totalNumBatches))
        self._trainingBatches = batchIndicies[0:numTrainBatches]
        self._testingBatches = batchIndicies[numTrainBatches:]
        return self

    def __preprocessFeatures(self,X):
        """ Apply a Standard Scaler to Inputs X """
        return X

    def __runLoadAndTrainSequence(self):
        """ Run data loading/training sequence """
        for batchIndex in self._trainingBatches:
            X,Y = self.__loadBatch(batchIndex)
            X = self.__preprocessFeatures(X)

            # Fit The Batch
            self._fitParams.batchSize = X.shape[0]
            self._fitParams.epochs = 4

            # Fit the Model
            trainingHistory = self._model.fit(
                x=X,
                y=Y,
                batchSize=self._fitParams.batchSize,
                epochs=self._fitParams.epochs,
                verbose=self._fitParams.verbose,
                callbacks=self._fitParams.callbacks,
                shuffle=self._fitParams.shuffle)
            self._trainingHistories.append(trainingHistory)

        # Done 
        return self

    def __runLoadAndTestSequence(self):
        """ Run data loading/testing sequence """
        return self

    def __exportExperimentDetails(self,iterCounter):
        """ Export the Details of the experient """
        return self


    def __resetState(self):
        """ Reset the State of the experiment in between iterations """
        self._seed * (2.0/3.0)
        self._model = None
        self._trainingBatches   = np.array([],dtype=np.int32)
        self._testingBatches    = np.array([],dtype=np.int32)
        self._trainingHistories.clear()
        return self

class MultilayerPerceptronExperiment(__BaseExperiment):
    """ Train + Test Multilater perceptron """
    pass
