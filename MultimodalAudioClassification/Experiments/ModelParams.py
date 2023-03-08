"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    Experiments
File:       ModelParams.py

Author:     Landon Buell
Date:       November 2022
"""

        #### IMPORTS ####

import numpy as np

        #### CLASS DEFINITIONS ####

class TensorFlowFitModelParams:
    """ Structure to store Params for fitting a TF Model """

    def __init__(self):
        """ Construction """
        self.batchSize  = None
        self.epochs     = 1
        self.verbose    = 'auto'
        self.callbacks  = []
        self.splitValidation = 0.0
        self.dataValidation = None
        self.shuffle    = True
        self.classWeight = None
        self.initialEpoch = 0

    def __del__(self):
        """ Destructor """
        pass


class ModelTrainingMetrics:
    """ Stores arrays of all of the metrics """

    def __init__(self):
        """ Constructor """
        self.iterCount  = 0
        self.loss       = np.array([],dtype=np.float64)
        self.accuracy   = np.array([],dtype=np.float64)
        self.precision  = np.array([],dtype=np.float64)
        self.recall     = np.array([],dtype=np.float64)

    def __del__(self):
        """ Destructor """
        self.iterCount  = None
        self.loss       = None
        self.accuracy   = None
        self.precision  = None
        self.recall     = None

    def updateWithBatchLog(self,batchLog):
        """ Update instance w/ data from batch Log """
        self.iterCount  += 1
        self.loss       = np.append(self.loss,batchLog['loss'])
        self.accuracy   = np.append(self.accuracy,batchLog['accuracy'])
        self.precision  = np.append(self.precision,batchLog['precision'])
        self.recall     = np.append(self.recall,batchLog['recall'])
        return self