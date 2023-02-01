"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    Experiments
File:       KFoldsCrossValidation.py

Author:     Landon Buell
Date:       January 2023
"""

        #### IMPORTS ####

import os
import numpy as np

import Experiments

        #### CLASS DEFINITIONS ####

class KFoldsCrossValidation:
    """ Execute a K-Folds X-Validation Strategy """

    def __init__(self,
                 runInfo,
                 outputPath,
                 numFolds,
                 modelLoaderCallback,
                 dataLoaderCallback,
                 seed=123456789):
        """ Constructor """
        self._runInfo       = runInfo
        self._outputPath    = outputPath 

        self._numFolds      = numFolds
        self._folds         = [None] * self._numFolds

        self._modelCallback = modelLoaderCallback
        self._dataCallback  = dataLoaderCallback

        self._seed          = seed
        np.random.seed(seed)


    def __del__(self):
        """ Destructor """
        pass

    # Getters and Setters

    # Public Interface

    def run(self):
        """ Run the K-Folds X-Validations """
        self.__initFoldExperiments()
        self.__assignBatchesToExperiments()

        return self

    # Private Interface

    def __initFoldExperiments(self):
        """ Initialize the Experiment within each fold """
        for ii in range(self._numFolds):
            iterSeed = (self._seed * ii)
            foldOutpath = os.path.join(self._outputPath,"fold{0}".format(ii))
            #foldModel = self._modelCallback.__call__(iterSeed)
            foldModel = None
            foldExperiment = Experiments.CrossValidationFoldExperiment(
                runInfo=self._runInfo,
                outputPath=foldOutpath,
                model=foldModel,
                dataloaderCallback=self._dataCallback,
                foldIndex=ii,
                numIters=1,
                seed=iterSeed)
            self._folds[ii] = foldExperiment
        return self

    def __assignBatchesToExperiments(self):
        """ Assign Batches to Each Experiment """
        numBatches = self._runInfo.getNumBatches()
        allBatches = np.arange(numBatches,dtype=np.int32)      
        allBatches = np.random.permutation(allBatches)
        splits = np.array_split(allBatches,self._numFolds)

        # Use the Splits to populate the Train/Test Batches in the Folds
        for ii in range(self._numFolds):
            copyOfSplits = splits[:]
            testBatches = copyOfSplits.pop(ii)
            trainBatches = copyOfSplits
            self._folds[ii].registerTestingBatches(testBatches)
            self._folds[ii].registerTrainingBatches(trainBatches)
       
        # All Done! 
        return self