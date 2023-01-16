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

import string

import sklearn.preprocessing as preprocessing


import PyToolsStructures
import PyToolsIO

        #### CONSTANTS ####

LETTERS_UPPER_CASE = list(string.ascii_uppercase)
LETTERS_LOWER_CASE = list(string.ascii_lowercase)

        #### CLASS DEFINITIONS ####

class StandardScaler:
    """ Apply Standard Scaling to Design Matrix """

    __NAME    = "standardScaler"
    
    def __init__(self,runInfo,outputPath):
        """ Constructor """
        self._runInfo       = runInfo
        self._outputPath    = outputPath

        totalNumSamples     = self._runInfo.getActualNumSamples()
        self._params        = [None] * PyToolsStructures.RunInformation.DEFAULT_NUM_PIPELINES
               
        self._sampleCounter = 0
        self._sampleData    = np.empty(shape=(totalNumSamples,),dtype=np.float32)

        if (os.path.isdir(self._outputPath) == False):
            os.makedirs(self._outputPath)

    def __del__(self):
        """ Destructor """
        pass

    # Getters and Setters

    def getOutFile(self,pipelineIndex):
        """ Get the output file for a particular pipeline """
        fileName = "pipeline{0}-{1}.txt".format(pipelineIndex,StandardScaler.__NAME) 
        result = os.path.join(self._outputPath,fileName)
        return result

    # Public Interface

    def fit(self,pipelines=None):
        """ Apply Standard scaler to all features """
        if (pipelines is None):
            pipelines = self._runInfo.getPipelinesInUse()
        return self.__fitHelper(pipelines)

    def exportParams(self):
        """ Export Learned Params to Disk for later use """

        return self

    # Private Interface

    def __fitHelper(self,pipelines):
        """ Private Helper for Fitting Data to the pipeline """
        for pipelineIndex in range(len(pipelines)):
            inUse = self.__pipelineIsInUse(pipelineIndex)
            if (inUse == False):
                continue
            # Process the pipeline if it is in Use
            self.__getParamsForPipeline(pipelineIndex)
        return self

    class __ScalerParams:
        """ Store the Params For all features within a single pipeline """
        
        def __init__(self,numFeatures):
            """ Constructor """
            self.means  = np.zeros(shape=(numFeatures,),dtype=np.float32)
            self.varis  = np.zeros(shape=(numFeatures,),dtype=np.float32)

        def __del__(self):
            """ Destructor """
            self.means = None
            self.varis = None

        @property
        def numFeatures(self):
             """ Get the number of features in this pipeline """
             return len(self.means)

    def __getParamsForPipeline(self,pipelineIndex):
        """ Fit All Features from this pipeline """
        pipelineSampleShape = self._runInfo.getSampleShapeOfPipeline(pipelineIndex)
        numFeatures = 1
        for axisSize in pipelineSampleShape:
            numFeatures = (numFeatures * axisSize)
        self._params[pipelineIndex] = StandardScaler.__ScalerParams(numFeatures)
        # Now Visit All Features
        for featureIndex in range(len(numFeatures)):
            self._sampleCounter = 0
            self.__getParamsForFeature(pipelineIndex,featureIndex,numFeatures)
            self.__setParamsForFeature(pipelineIndex,featureIndex)           
        # Export the Means + Varis for All Features
        self.__exportParamsForFeatures(pipelineIndex)
        return self

    def __getParamsForFeature(self,pipelineIndex,featureIndex,numFeatures):
        """ Fit feature from specific Pipeline """
        rawDataPath = self._runInfo.getOutputPath()
        batchSizes  = self._runInfo.getBatchSizes()
        pipelineIndetifier = LETTERS_UPPER_CASE[pipelineIndex]
        for batchIndex,batchSize in enumerate(batchSizes):
            pathX = PyToolsStructures.getDesignMatrixDataPath(
                rawDataPath,batchIndex,pipelineIndetifier)
            pathY = PyToolsStructures.getDesignMatrixLabelsPath(
                rawDataPath,batchIndex)
            # Load in Design Matrix from batch
            designMatrix = PyToolsStructures.DesignMatrix.deserialize(
                pathX,pathY,batchSize,numFeatures)
            featureFromAllSamples = designMatrix.getFeatures()[:,featureIndex]
            # Populate Larger Data Array w/ Samples
            for ii in range(batchSize):
                self._sampleData[self._sampleCounter] = featureFromAllSamples[ii]
                self._sampleCounter += 1
        # All Done!
        return self

    def __setParamsForFeature(self,pipelineIndex,featureIndex):
        """ Store the Mean + Vari Data for this Pipeline/Feature """   
        self._params[pipelineIndex].means[featureIndex] = np.mean(self._sampleData)
        self._params[pipelineIndex].varis[featureIndex] = np.var(self._sampleData)
        return self

    def __exportParamsForFeatures(self,pipelineIndex):
        """ Return Pipeline Index """
        pipelineParams  = self._params[pipelineIndex]
        if (pipelineParams is None):
            return self
        outputFile = self.getOutFile(pipelineIndex)
        outputStream    = open(outputFile,"w")
        for ii in range(pipelineParams.numFeatures):
            line = "{0}\t{1}\n".format(
                pipelineParams.means[ii],
                pipelineParams.varis[ii])
            outputStream.write(line)
        outputStream.close()
        return self


class StandardScalerWrapper:
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
        writer = StandardScalerWrapper.StandardScalerSerializer(self,exportPath)
        success = writer.call()
        return success

    @staticmethod
    def deserialize(importPath):
        """ Read Params from Text File """
        reader = StandardScalerWrapper.StandardScalerDeserializer(importPath)
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
            self._data = StandardScalerWrapper(numFeatures)
            self._data._scaler.n_samples_seen_  = numSamples
            self._data._scaler.mean_            = means
            self._data._scaler.var_             = varis
            self._data._scaler.scale_           = scale

            return self._data





