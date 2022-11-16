"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    Models
File:       Strategies.py

Author:     Landon Buell
Date:       November 2022
"""

    #### IMPORTS ####

import os
import sys

import tensorflow as tf

import PyToolsStructures

    #### CLASS DEFINITIONS ####

class Strategy:
    """ Parent Class for Execution Strategies """

    def __init__(self,featurePath,processedPath,outPath,
                 model,batchesToUse=None,numIters=1,seed=0):
        """ Constructor """

        self._runPath       = featurePath   # Path to 'raw' Feature Data
        self._paramsPath    = processedPath # Path to 'processed' Feature data
        self._outputPath    = outPath       # path to send results to
        self._model         = model         # Model to work with
   
        self._runInfo       = PyToolsStructures.RunInfo.deserialize(featurePath)
        if (batchesToUse is None):
            self._batchesToUse = [x for x in len(self._runInfo.getNumBatches())]
 
        self._numIters      = numIters
        self._randomSeed    = seed

    def __del__(self):
        """ Destructor """

    # Getters and Setters

    def getFeaturesPath(self):
        """ Get the Path to the Feature Data """
        return self._runPath

    def getProcessedPath(self):
        """ Get the Path to the processed params """
        return self._paramsPath

    def getOutputPath(self):
        """ Get the Path to the outputs """
        return self._outputPath

    def getRunInfo(self):
        """ Get the Run Info Structure """
        return self._runInfo

    def getBatchesToUse(self):
        """ Get list of batch Indexes to use """
        return self._batchesToUse

    def getNumIters(self):
        """ Get the number of iterations to run """
        return self._numIters

    def getRandomSeed(self):
        """ Get the Random Seed """
        return self._randomSeed

    # Public Interface

    def call(self):
        """ Invoke this strategy """


        return self


    # Protected Interface


class TrainStrategy(Strategy):
    """ Class to Train a Model """

    def __init__(self,featurePath,processedPath,outPath,
                 model,batchesToUse=None,numIters=1,seed=0):
        """ Constructor """
        super().__init__(featurePath,processedPath,outPath,model,batchesToUse,numIters,seed)

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Getters and Setters

    # Public Interface

    # Protected Interface
    
