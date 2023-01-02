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



import numpy as np
import tensorflow as tf

import PyToolsStructures

    #### CLASS DEFINITIONS ####

class Strategy:
    """ Parent Class for Execution Strategies """

    INIT_NUM_DESIGN_MATRIX_SAMPLES = 1024

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
        self._numIters          = numIters

        self._randomSeed        = seed
        np.random.seed(self._seed)

        self._batchesRemaining  = range(self._runInfo.getNumBatches())
        self._batchesAtOnce     = min(batchesAtOnce,self._runInfo.getNumBatches())

        self._modeA         = loadA
        self._modeB         = loadB

        self._designMatrixA     = PyToolsStructures.DesignMatrix(
            Strategy.INIT_NUM_DESIGN_MATRIX_SAMPLES,
            self._runInfo.getMatrixShape(0))
        self._designMatrixB     = PyToolsStructures.DesignMatrix(
            Strategy.INIT_NUM_DESIGN_MATRIX_SAMPLES,
            self._runInfo.getMatrixShape(1))

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
        """ VIRTUAL: Invoke this strategy """

      

        return self


    # Protected Interface

    def resetRemainingBatches(self):
        """ Reset the available batches for each loop """
        self._batchesRemaining = range(self._runInfo.getNumBatchs())
        return self

    def populateDesignMatrices(self):
        """ Populate Design Matrices """
        batchesToLoad = self.drawRandomBatchIndices()
        batchSizes = self._runInfo.getBatchSizes()
        numSamples = 0
        for index in len(batchesToLoad):
            numSamples += batchSizes[batchesToLoad]
        # Now allocate design matices
        if (self._modeA == True):
            self._designMatrixA = self._designMatrixA.setNumSamples(numSamples)
        if (self._modeB == True):
            self._designMatrixB = self._designMatrixB.setNumSamples(numSamples)
        # Now Populate the design matrices
        designMatrixSampleIndex = 0
        for ii in range(len(batchesToLoad)):
            batchSize = batchesToLoad[ii]
            matrices = self._runInfo.loadBatch(batchesToLoad[ii])
            for jj in range(len()):
                if (self._modeA == True):
                    self._designMatrixA[designMatrixSampleIndex] = matrices[0][jj]
                if (self._modeB == True):
                    self._designMatrixB[designMatrixSampleIndex] = matrices[1][jj]
                designMatrixSampleIndex += 1
        # The design matrices are populated with samples in chosen batches
        return self

    def drawRandomBatchIndices(self):
        """ Draw 4 batches at time """
        self._batchesRemaining = np.random.permute(self._batchesRemaining)
        drawnBatches = []
        for ii in range(self._batchesAtOnce):
            # Check if any batches remain
            if (len(self._batchesRemaining) == 0):
                break
            # Otherwise, drawn again
            draw = self._batchesRemaining.pop()
            drawnBatches.append(draw)
        return drawnBatches

    def validateModelMatchesRunInfo(self):
        """ Confirm that the model is compatible with the data """



class TrainStrategyMultimodal(Strategy):
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
        
        for ii in range(self._numIters):    
            self.printIterMessage(ii)
            self.resetRemainingBatches()

            while (len(self._batchesRemaining) > 0):
                # While there are batches
                self.populateDesignMatrices()


            

        return self

    # Protected Interface
    
    def printIterMessage(self,iter):
        """ Print Iteration Message """
        msg = "\tTraining iteration #{0}".format(iter)
        print(msg)
        return self

    def fitModel(self):
        """ fit the Model """
        X = (None,None)
        Y = (None,None)
        if (self._modeA == True):
            X[0] = self._designMatrixA.getFeatures()
        if (self._modeB == True):

        # Train the model
        history = self._model.fit(
            )

        return self