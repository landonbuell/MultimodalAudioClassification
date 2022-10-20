"""
Repository:     MultimodalAudioClassification
Solution:       MultimodalAudioClassification
Project:        CommonToolsPy
File:           PyToolsStructures.py
 
Author:         Landon Buell
Date:           June 2022
"""

        #### IMPORTS ####

import os
import numpy as np

import PyToolsStructures
import PyToolsIO

        #### CLASS DEFINITIONS ####

class StandardScaler:
    """ Apply Standard Scaling to Design Matrix """

    def __init__(self,numFeatures):
        """ Constructor """
        self._samplesSeen   = 0
        self._numFeatures   = numFeatures
        self._means         = np.zeros(shape=(numFeatures,),dtype=np.float32)
        self._varis         = np.ones(shape=(numFeatures,),dtype=np.float32)

    def __del__(self):
        """ Destructor """
        pass

    # Getters and Setters

    def getNumSampleSeen(self):
        """ Return the number of samples seen """
        return self._samplesSeen

    def getIsFit(self):
        """ Return T/F if scaler has been fit """
        return bool(self._samplesSeen)

    def getNumFeatures(self):
        """ Retutn the number of features seen """
        return self._numFeatures

    def getMeans(self):
        """ Get the Means for each feature """
        return self._means

    def getVaris(self):
        """ Get the variances for each feature """
        return self._varis

    # Public Interface

    def fit(self,designMatrix):
        """ Fit the Design matrix """
        numSamples = designMatrix.getNumSamples()
        numFeatures = designMatrix.getNumFeatures()
        if (self._numFeatures != numFeatures):
            msg = "WARNING: StandardScaler expected {0} features but got {1}".format(self._numFeatures,numFeatures)
            raise RuntimeError(msg)
        self._means = ((self._samplesSeen * self._means) + (numSamples * designMatrix.means())) / 2.0
        self._varis = ((self._samplesSeen * self._varis) + (numSamples * designMatrix.variances())) / 2.0;
        self._samplesSeen += numSamples
        return self

    def call(self,designMatrix):
        """ Apply Scale factor to design matrix """
        if (self.getIsFit() == False):
            msg = "WARNING: StandardScaler not fit - returning unchanged design matrix"
            print(msg)
            return designMatrix
        # Is Fit
        X = designMatrix.getFeatures()
        X = (X - self._means) / np.sqrt(self._varis)
        designMatrix.setFeatures(X)
        means = designMatrix.means()
        varis = designMatrix.variances()
        return designMatrix

    def serialize(self,exportPath):
        """ Write Params to Text File """
        return self

    @staticmethod
    def deserialize(self,exportPath):
        """ Read Params from Text File """
        return None

    # Private Interface




