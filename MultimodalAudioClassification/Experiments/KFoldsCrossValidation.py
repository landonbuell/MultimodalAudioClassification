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
                 experiment,
                 numFolds,
                 seed=123456789):
        """ Constructor """
        self._experiment = experiment 
        self._headOutputPath = self._experiment.getOutputPath()
        self._numFolds      = numFolds
        self._folds         = [None] * self._numFolds
        self._seed          = seed
        np.random.seed(seed)


    def __del__(self):
        """ Destructor """
        pass

    # Getters and Setters

    def getRunInfo(self):
        """ Get the RunInfo Struct """
        return self._experiment.getRunInfo()

    def getOutputPath(self):
        """ Get the K-Folds output Path """
        return self._experiment.getOutputPath()

    # Public Interface

    def run(self):
        """ Run the K-Folds X-Validations """
        self.__initializeFolds()

        for ii in range(self._numFolds):
            self.__overrideExperimentOutputPath(ii)
            self.__registerBatchesWithExperiment(ii)

            # Execute the Experiment
            self._experiment.run()

        return self

    # Private Interface

    def __initializeFolds(self) -> None:
        """ Determine which batches go with which folds """
        batches = np.arange(self.getRunInfo().getNumBatches(),dtype=np.int16)
        np.random.shuffle(batches)
        for ii in range(self._numFolds):
            self._folds[ii] = []

        for ii,batch in enumerate(batches):
            foldIndex = np.mod(ii,self._numFolds)
            self._folds[foldIndex].append( batches[ii] )

        return None

    def __overrideExperimentOutputPath(self,foldIndex: int) -> None:
        """ Override the output path for the current experiment """
        foldIndexText = "fold{0}".format(foldIndex)
        newOutputPath = os.path.join(self._headOutputPath,foldIndexText)
        self._experiment.setOutputPath(newOutputPath)
        return None

    def __registerBatchesWithExperiment(self,foldIndex: int) -> None:
        """ Set the training + Testing Batches w/ """
        for ii in range(self._numFolds):
            if (ii == foldIndex):
                self._experiment.registerTestingBatches( self._folds[ii] )
            else:
                self._experiment.registerTrainingBatches( self._folds[ii] )
        # All Batches Registered
        return self
            




