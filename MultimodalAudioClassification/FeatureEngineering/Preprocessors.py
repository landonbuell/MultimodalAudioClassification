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
import sklearn.preprocessing as preprocessing
from sklearn.utils.extmath import _deterministic_vector_sign_flip

import PyToolsStructures
import PyToolsIO

        #### CLASS DEFINITIONS ####

class StandardScaler:
    """ Apply Standard Scaling to Design Matrix """

    def __init__(self,numFeatures):
        """ Constructor """
        self._scaler = preprocessing.StandardScaler(copy=False)


    def __del__(self):
        """ Destructor """
        pass

    # Getters and Setters

    def getNumSampleSeen(self):
        """ Return the number of samples seen """
        return self._scaler.n_samples_seen_

    def getIsFit(self):
        """ Return T/F if scaler has been fit """
        return bool(self._scaler.n_samples_seen_)

    def getNumFeatures(self):
        """ Retutn the number of features seen """
        return len(self._scaler.n_features_in_)

    def getMeans(self):
        """ Get the Means for each feature """
        return self._scaler.mean_

    def getVaris(self):
        """ Get the variances for each feature """
        return self._scaler.var_

    # Public InterfaceZ

    def fit(self,designMatrix):
        """ Fit the Design matrix """   
        self._scaler.fit(designMatrix.getFeatures())
        return self

    def call(self,designMatrix):
        """ Apply Scale factor to design matrix """
        X = self._scaler.transform(designMatrix.getFeatures())
        designMatrix.setFeatures(X)
        return X

    def serialize(self,exportPath):
        """ Write Params to Text File """
        return self

    @staticmethod
    def deserialize(self,exportPath):
        """ Read Params from Text File """
        return None

    # Private Interface




