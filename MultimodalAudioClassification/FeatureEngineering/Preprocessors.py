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
        return self._scaler.n_features_in_

    def getMeans(self):
        """ Get the Means for each feature """
        return self._scaler.mean_

    def getVaris(self):
        """ Get the variances for each feature """
        return self._scaler.var_

    def getScale(self):
        """ Get the scale factor for each feature """
        return self._scaler.scale_

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
        writer = StandardScaler.StandardScalerSerializer(self,exportPath)
        success = writer.call()
        return success

    @staticmethod
    def deserialize(importPath):
        """ Read Params from Text File """
        reader = StandardScaler.StandardScalerDeserializer(importPath)
        result = reader.call()
        return result

    # Private Interface

    class StandardScalerSerializer(PyToolsIO.Serializer):
        """ Class to Serialize StandardScaler """

        def __init__(self,data,path):
            """ Constructor """
            super().__init__(data,path)

        def __del__(self):
            """ Destructor """
            super().__del__()

        def call(self):
            """ Run the serializer """
            self._outFileStream = open(self._outputPath,"w")

            # Write Num Samples Seen
            numSamples = str(self._data.getNumSampleSeen())
            numFeatures = str(self._data.getNumFeatures())
            self._outFileStream.write( numSamples + " " + numFeatures + "\n")

            # Write Means + Variance
            means = PyToolsIO.Serializer.listToString(self._data.getMeans())
            varis = PyToolsIO.Serializer.listToString(self._data.getVaris())
            scale = PyToolsIO.Serializer.listToString(self._data.getScale())
            self._outFileStream.write( means + "\n" )
            self._outFileStream.write( varis + "\n" )
            self._outFileStream.write( scale + "\n" )

            self._outFileStream.close()
            return True

    class StandardScalerDeserializer(PyToolsIO.Deserializer):
        """ Class to Deserialize StandardScaler """

        def __init__(self,path):
            """ Constructor """
            super().__init__(path)

        def __del__(self):
            """ Destructor """
            super().__del__()

        def call(self):
            """ Run the Deserializer """
            fileContents = self._inFileStream.readlines()
            tokensLine0 = fileContents[0].split()
            numSamples = int(tokensLine0[0])
            numFeatures = int(tokensLine0[1])
            means = PyToolsIO.Deserializer.stringToList(fileContents[1],delimiter=",",outType=float)
            varis = PyToolsIO.Deserializer.stringToList(fileContents[2],delimiter=",",outType=float)
            scale = PyToolsIO.Deserializer.stringToList(fileContents[3],delimiter=",",outType=float)

            # Make the Standard Scaler Instance
            self._data = StandardScaler(numFeatures)
            self._data._scaler.n_samples_seen_  = numSamples
            self._data._scaler.mean_            = means
            self._data._scaler.var_             = varis
            self._data._scaler.scale_           = scale

            return self._data





