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
                 experiment,
                 seed=123456789):
        """ Constructor """
        self._runInfo       = runInfo
        self._outputPath    = outputPath 

        self._numFolds      = numFolds
        self._folds         = [None] * self._numFolds

        self._currentExeriment = experiment

        self._seed          = seed
        np.random.seed(seed)

        self.__validateRunInfo()

    def __del__(self):
        """ Destructor """
        pass

    # Getters and Setters


    # Public Interface

    def run(self):
        """ Run the K-Folds X-Validations """
        self.__initializeFolds()

        for ii in range(self._numFolds):
            self.__overrideExperimentOutputPath(ii)
            self.__registerBatchesWithExperument(ii)

        return self

    # Private Interface

    def __validateRunInfo(self) -> None:
        """ Make sure that the experiment runInfo is the same as the folds info """
        if (self._currentExperiment.getRunInfo() is not self._runInfo):
            # Must be the same runInfo instance 
            errMsg = "Got two different runInfo structs for current experiment and folds experiments"
            raise RuntimeError(errMsg)
        return None

    def __initializeFolds(self) -> None:
        """ Determine which batches go with which folds """
        batches = range(self._runInfo().getNumBatches())
        for ii in range(self._numFolds):
            self._folds[ii] = list()

        # Build up the batches that go into each fold
        while (len(batches > 0)):
            for ii in range(self._numFolds):
                batchIndex = batches.pop()
                self._folds[ii].append(batchIndex)
            # End for-loop
        # End while-loop

        return None

    def __overrideExperimentOutputPath(self,foldIndex: int) -> None:
        """ Override the output path for the current experiment """
        oldOutputPath = self._currentExeriment.getOutputPath()
        foldIndexText = "fold{0}".format(foldIndex)
        newOutputPath = os.path.join(oldOutputPath,foldIndexText)
        self._currentExeriment.setOutputPath(newOutputPath)
        return None

    def __registerBatchesWithExperument(self,foldIndex: int) -> None:
        """ Set rhe training + Testing Batches w/ """





