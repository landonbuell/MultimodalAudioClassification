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

    def __init__(self,
                 runInfo,
                 outputPath,
                 model,
                 numIters=1,
                 seed=0,
                 batchesAtOnce=64,
                 loadA=True,
                 loadB=True):
        """ Constructor """

        self._runInfo           = runInfo     # load data from here
        self._outputPath        = outputPath    # path to send results to
        self._model             = model         # Model to work with
 
        self._numIters      = numIters
        self._randomSeed    = seed
        self._batchIndex    = 0
        self._batchesAtOnce = min(batchesAtOnce,self._runInfo.getNumBatches())

        self._modeA         = loadA
        self._modeB         = loadB

        self._designMatrixA     = None
        self._designMatrixB     = None

    def __del__(self):
        """ Destructor """
        self._runInfo       = None
        self._model         = None
        self._designMatrixA = None
        self._designMatrixB = None

    # Getters and Setters

    def getRunInfo(self):
        """ Get the RunInfo Structure """
        return self._runInfo

    def getInputPath(self):
        """ Get the Path to the Features """
        return self._runInfo.getOutputPath()

    def getOutputPath(self):
        """ Get the Path to the outputs """
        return self._outputPath

    def getModel(self):
        """ Get the Model being worked on """
        return self._model

    def getNumIters(self):
        """ Get the number of iterations to run """
        return self._numIters

    def getRandomSeed(self):
        """ Get the Random Seed """
        return self._randomSeed

    # Public Interface

    def call(self):
        """ Invoke this strategy """

        self.populateDesignMatrices()

        return self


    # Protected Interface

    def populateDesignMatrices(self):
        """ Populate Design Matrices """

        batchSizes      = self._runInfo.getBatchSizes()
        batchStartIndex = self._batchIndex
        batchStopIndex  = self._batchIndex + self._batchesAtOnce
        if (len(batchSizes) < batchStopIndex):
            batchStopIndex = (len(batchSizes) - 1)
        batchesToLoad = range(batchStartIndex,batchStopIndex)
        numSamplesInBatchSet = sum(batchSizes[batchesToLoad])

        # Build Design Matrices Based off the Number of samples
        self._designMatrixA = PyToolsStructures.DesignMatrix(numSamplesInBatchSet,self._runInfo.getMatrixShape(0))
        self._designMatrixB = PyToolsStructures.DesignMatrix(numSamplesInBatchSet,self._runInfo.getMatrixShape(1))

        # Load in Design Matrix
        for ii in range(batchesToLoad):
            # Message To Console
            msg = "\tLoading Batch #{0}".format(self._batchIndex)
            print(msg)
            
            matrices = self._runInfo.loadBatch(
                self._batchIndex,self._loadA,self._loadB)

        # Increment Index Counter + Terminate
        self._batchIndex = batchStopIndex
        return self


class TrainStrategy(Strategy):
    """ Class to Train a Model """

    def __init__(self,
                 runInfo,
                 outputPath,
                 model,
                 numIters=1,
                 seed=0):
        """ Constructor """
        super().__init__(runInfo,outputPath,model,numIters,seed)
        self._trainHistory = []

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Getters and Setters

    def getTrainHistory(self):
        """ Return the list of training history """
        return self._trainHistory

    # Public Interface

    def call(self):
        """ Invoke this strategy """

        # Load in a Data Subset


        return self

    # Protected Interface
    
    def printIterMessage(self,iter):
        """ Print Iteration Message """
        msg = "\tTraining iteration #{0}".format(iter)
        print(msg)
        return self

    def fitModel(self):
        """ fit the Model """


        return self