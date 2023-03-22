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

class Preprocessor:
    """ Base class for all preprocessors """

    def __init__(self,runInfo,outputFolder):
        """ Constructor """
        self._runInfo       = runInfo
        self._outputPath    = os.path.join(runInfo.getOutputPath(),outputFolder)

        if (os.path.isdir(self._outputPath) == False):
            os.makedirs(self._outputPath)

    def __del__(self):
        """ Destructor """

    # Public Interface

    def getRunInfo(self):
        """ Get ref to run info """
        return self._runInfo

    def getOutputPath(self) -> str:
        """ Get output path """
        self._outputPath

    # Protected Interface

    def _getNumFeaturesInPipeline(self,pipelineIndex):
        """ Get the total number of features in a pipeline """
        numFeatures = 1
        pipelineShape = self._runInfo.getSampleShapeOfPipeline(pipelineIndex)
        for ii in range(len(pipelineShape)):
            numFeatures *= pipelineShape[ii]
        return numFeatures


class CustomStandardScaler(Preprocessor):
    """ Apply Standard Scaling to Design Matrix """

    __NAME    = "standardScaler"
    
    def __init__(self,runInfo):
        """ Constructor """
        super().__init__(runInfo,CustomStandardScaler.__NAME)
      
        self._params            = [None] * PyToolsStructures.RunInformation.DEFAULT_NUM_PIPELINES
        self._featuresAtOnce    = 1024

        totalNumSamples         = self._runInfo.getActualNumSamples()
        sampleDataShape         = (totalNumSamples,self._featuresAtOnce)
               
        self._sampleCounter = 0
        self._sampleData    = np.empty(shape=sampleDataShape,dtype=np.float32)


    def __del__(self):
        """ Destructor """
        pass

    # Getters and Setters

    def getOutFile(self,pipelineIndex):
        """ Get the output file for a particular pipeline """
        fileName = "pipeline{0}.txt".format(pipelineIndex) 
        result = os.path.join(self._outputPath,fileName)
        return result

    def getParams(self,pipelineIndex):
        """ Get Parameters for pipeline """
        return self._params[pipelineIndex]

    # Public Interface

    def fit(self,pipelines=None):
        """ Apply Standard scaler to all features """
        if (pipelines is None):
            pipelines = self._runInfo.getPipelinesInUse()
        return self.__fitHelper(pipelines)

    def applyFitToMatrix(self,designMatrix,pipelineIndex):
        """ Apply Fit Params to Design Matrix """
        fitParams = self._params[pipelineIndex]
        if (fitParams.numFeatures != designMatrix.shape[-1]):
            msg = "Expected {0} features but got {1}".format(fitParams.numFeatures,designMatrix.shape[-1])
            raise RuntimeError(msg)
        designMatrix = (designMatrix -  fitParams.means) / np.sqrt(fitParams.varis)
        means = np.mean(designMatrix,axis=0)
        varis = np.std(designMatrix,axis=0)
        return self

    def exportParams(self,pipelineIndex):
        """ Export Learned Params to Disk for later use """
        params = self._params[pipelineIndex]
        outStream = open(self.getOutFile(pipelineIndex),"w")
        for ii,(m,v) in enumerate(zip(params.means,params.varis)):
            line = "{0:<16}{1:<32}{2:<32}\n".format(ii,m,v)
            outStream.write(line)
        outStream.close()
        return self

    def loadParams(self,pipelineIndex):
        """ Import Learned Params to Disk for later use """
        numFeatures = self._getNumFeaturesInPipeline(pipelineIndex)
        newParams = CustomStandardScaler.__ScalerParams(numFeatures)
        inputStream = open(self.getOutFile(pipelineIndex),"r")
        while True:
            line = inputStream.readline()
            if not line:
                break
            lineTokens = line.split()
            index = int(lineTokens[0])
            newParams.means[index] = np.float32(lineTokens[1])
            newParams.varis[index] = np.float32(lineTokens[2])
        inputStream.close()
        self._params[pipelineIndex] = newParams
        return self

    # Private Interface

    def __fitHelper(self,pipelines):
        """ Private Helper for Fitting Data to the pipeline """
        for pipelineIndex in pipelines:
            inUse = self._runInfo.getIsPipelineInUse(pipelineIndex)
            if (inUse == False):
                continue
            # Process the pipeline if it is in Use
            self.__processPipeline(pipelineIndex)
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

    def __processPipeline(self,pipelineIndex):
        """ Process all the Features in this pipeline """
        numFeaturesInPipeline = self._getNumFeaturesInPipeline(pipelineIndex)
        self._params[pipelineIndex] = CustomStandardScaler.__ScalerParams(numFeaturesInPipeline)

        featureStartIndex = 0
        featureStopIndex = min([featureStartIndex + self._featuresAtOnce,numFeaturesInPipeline])

        #numLoops = np.ceil(self._featuresAtOnce/numFeaturesInPipeline)
        #self.__printProcessingFeaturesMessage(pipelineIndex,featureStartIndex,featureStopIndex)

        # Process a Group of Features
        while (featureStartIndex < numFeaturesInPipeline):
            self.__printProcessingFeaturesMessage(pipelineIndex,featureStartIndex,featureStopIndex)
            featuresToProcessMask = np.arange(featureStartIndex,featureStopIndex,dtype=np.int16)  
            
            self._sampleData = np.empty(
                shape=(self._runInfo.getActualNumSamples(),len(featuresToProcessMask),),
                       dtype=np.float32)
           
            self.__storeFeatureGroup(
                featuresToProcessMask,
                pipelineIndex,
                numFeaturesInPipeline)
            self.__processFeatureGroup(
                featuresToProcessMask,
                pipelineIndex)
            
            # increment Feature To Process Mask
            featureStartIndex += self._featuresAtOnce
            featureStopIndex = min([featureStartIndex + self._featuresAtOnce,numFeaturesInPipeline])
            
        # All Groups of Features Are processed
        self.exportParams(pipelineIndex)
        return self

    def __storeFeatureGroup(self,featureMask,pipelineIndex,numFeatures):
        """ Store all Samples from a Group of Features """
        batchSizes = self._runInfo.getBatchSizes()
        outputPath = self._runInfo.getOutputPath()
        nameX = lambda x,y  : "batch{0}x-pipeline{1}.bin".format(x,y)
        nameY = lambda x    : "batch{0}y.bin".format(x)
        for ii,numSamples in enumerate(batchSizes):

            pipelineIdentifier = LETTERS_UPPER_CASE[pipelineIndex]
            pathX = os.path.join(outputPath,nameX(ii,pipelineIdentifier))
            pathY = os.path.join(outputPath,nameY(ii))

            # Load + Store Design Matrix
            matrix = PyToolsStructures.DesignMatrix.deserialize(
                pathX,pathY,numSamples,(numFeatures,))
            self.__storeDesignMatrix(matrix,featureMask)

        # All Features are stored   
        self._sampleCounter = 0   
        return self

    def __processFeatureGroup(self,featureMask,pipelineIndex):
        """ Process the group of features """
        allMeans = np.mean(self._sampleData,axis=0)
        allVaris = np.var(self._sampleData,axis=0)

        for ii,featureIndex in enumerate(featureMask):
            self._params[pipelineIndex].means[featureIndex] = allMeans[featureIndex]
            self._params[pipelineIndex].varis[featureIndex] = allVaris[featureIndex]

        return self
        
    def __storeDesignMatrix(self,designMatrix,featureMask):
        """ Store subset of Features from Design Matrix """
        X = designMatrix.getFeatures()[:,featureMask]
        for ii in range(len(X)):
            self._sampleData[self._sampleCounter] = X[ii]
            self._sampleCounter += 1
        return self

    def __printProcessingFeaturesMessage(self,pipelineID,start,stop):
        """ Print message to console showing what features are being proccessed """
        msg = "\tProcessing Pipeline {0}: Features {1} -> {2}"
        print(msg.format(pipelineID,start,stop))
        return None
         
class StandardScalerWrapper(Preprocessor):
    """ Apply Standard Scaling to Design Matrix """

    def __init__(self,runInfo):
        """ Constructor """
        super().__init__(runInfo,"standardScalerWrapper")
        self._runInfo = runInfo
        self._scalers = [None] * PyToolsStructures.RunInformation.DEFAULT_NUM_PIPELINES

        pipelinesInUse = self._runInfo.getPipelinesInUse()
        for pipelineIndex in pipelinesInUse:
            self._scalers[pipelineIndex] = preprocessing.StandardScaler(copy=False)
        
    def __del__(self):
        """ Destructor """
        pass

   # Getters and Setters

    def getOutFile(self,pipelineIndex):
        """ Get the output file for a particular pipeline """
        fileName = "pipeline{0}.txt".format(pipelineIndex) 
        result = os.path.join(self._outputPath,fileName)
        return result

    def getParams(self,pipelineIndex):
        """ Get Parameters for pipeline """
        return self._params[pipelineIndex]

    # Public Interface

    def fit(self,pipelines=None):
        """ Apply Standard scaler to all features """
        if (pipelines is None):
            pipelines = self._runInfo.getPipelinesInUse()
        return self.__fitHelper(pipelines)

    def applyFitToMatrix(self,designMatrix,pipelineIndex):
        """ Apply Fit Params to Design Matrix """
        scaler = self._scalers[pipelineIndex]
        designMatrixScaled = designMatrix.copy()
        scaler.transform(designMatrixScaled)
        return designMatrixScaled

    def exportParams(self,pipelineIndex):
        means = self._scalers[pipelineIndex].mean_
        varis = self._scalers[pipelineIndex].var_
        outStream = open(self.getOutFile(pipelineIndex),"w")
        for ii,(m,v) in enumerate(zip(means,varis)):
            line = "{0:<16}{1:<32}{2:<32}\n".format(ii,m,v)
            outStream.write(line)
        outStream.close()

        return self

    def loadParams(self,pipelineIndex):
        """ Import Learned Params to Disk for later use """
        numFeatures = self._getNumFeaturesInPipeline(pipelineIndex)
        means = np.zeros(shape=(numFeatures,),dtype=np.float32)
        varis = np.zeros(shape=(numFeatures,),dtype=np.float32)

        inputStream = open(self.getOutFile(pipelineIndex),"r")
        while True:
            line = inputStream.readline()
            if not line:
                break
            lineTokens = line.split()
            index = int(lineTokens[0])
            means[index] = np.float32(lineTokens[1])
            varis[index] = np.float32(lineTokens[2])
        inputStream.close()
        
        newScaler = preprocessing.StandardScaler(copy=False)
        newScaler.mean_  = means
        newScaler.var_   = varis
        newScaler.scale_ = np.sqrt(varis)
        newScaler.n_features_in_ = numFeatures
        newScaler.n_samples_seen_ = self._runInfo.getActualNumSamples()
        self._scalers[pipelineIndex] = newScaler
        return self

    # Private Interface

    def __fitHelper(self,pipelines):
        """ helper for fitting """
        self.__partialFitAllSamples(pipelines)
        self.__exportParams()
        return self

    def __partialFitAllSamples(self,pipelines):
        """ Apply partial fit to all samples in dataset """
        numBatches = self._runInfo.getNumBatches()
        for batchIndex in range(numBatches):
            msg = "Partial Fitting batch {0}/{1} ...".format(batchIndex,numBatches)
            print(msg)
            designMatrices = self._runInfo.loadSingleBatchFromPipelines(batchIndex,pipelines)

            for ii,matrix in enumerate(designMatrices):
                if (matrix is None):
                    continue

                # Extract Design Matrix and Apply Partial Fit
                X = matrix.getFeatures()
                self._scalers[ii].partial_fit(X)
        return self

    def __exportParams(self):
        """ Export Parameters """
        for ii in range(len(self._scalers)):
            if (self._scalers[ii] is None):
                continue
            # Export
            self.exportParams(ii)
        return



