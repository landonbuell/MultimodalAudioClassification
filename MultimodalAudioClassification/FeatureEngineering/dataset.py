"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureEngineering
    File:       dataset.py
    Classes:    Dataset

    Author:     Landon Buell
    Date:       April 2024
"""

        #### IMPORTS ####

import os
import threading
import datetime

import numpy as np

import sessionInformation
import designMatrix
import datasetPipeline
import datasetLoader

        #### CLASS DEFINITIONS ####

class Dataset:
    """ 
        Represents a dataset that can be loaded for later use 
    """

    SAMPLES_PER_FILE = 1000     # Fetch from RunInfo Later

    def __init__(self,
                 rootPath: str,
                 seed=123456789,
                 verbose=True):
        """ Constructor """
        self._rootPath      = rootPath
        self._verbose       = verbose
        self.__validateRootPath()

        # Dataset Members
        self._runInfo       = sessionInformation.RunInfo()
        self._classInfo     = sessionInformation.ClassInfoDatabase()
        self._loader        = datasetLoader.DatasetLoader(self)
        self._pipelines     = []

        # If None is provided, use all pipelines, otherwise just use x
        self._filterPipelines = None

        # Load in all of the internal information
        self.__loadClassInfo()
        self.__loadPipelines()
        self._loader.populate()
        self._loader.shuffle(seed)

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getRoot(self) -> str:
        """ Return the root of the dataset """
        return self._rootPath

    def getNumPipelines(self) -> int:
        """ Return the number of pipelines """
        return len(self._pipelines)

    def getNumSamples(self) -> int:
        """ Return the number of samples """
        return self._loader.getSize()

    def getPipelineNames(self) -> list:
        """ Get a list of all names for each pipeline line """
        return [x.getName() for x in self._pipelines]

    def getPipeline(self,pipelineIndex: int) -> datasetPipeline.DatasetPipeline:
        """ Return the pipeline at the index """
        return self._pipelines[pipelineIndex]

    def getNumFeatures(self,pipelineIndex: int) -> int:
        """ Return the number of features in the chosen pipeline """
        return self._pipelines[pipelineIndex].getNumFeatures()

    def getClassInfo(self, classIndex: int) -> object:
        """ Return info about a class """
        return self._classInfo[classIndex]

    def getIndexFromName(self, name: str) -> int:
        """ Get the pipeline index from a provided name. -1 if not found """
        index = -1
        for ii,pipeline in enumerate(self._pipelines):
            if (pipeline.getName() == name):
                index = ii
                break
        return index

    # Public Interface - Load Multiple

    def loadSamples(self,
             sampleIDs: np.ndarray,
             pipelines=None) -> designMatrix.MultimodalDesignMatrix:
        """ Draw samples corresponding to the provided index """
        pipelines = self.__filterPipelines(pipelines)
        toLoad = sampleIDs
        designMatrix = self.__loadDesignMatrixFromPipelines(
            sampleIDs=toLoad,
            fromPipelines=pipelines)
        return designMatrix

    def loadAllFromClass(self,
             classIndex: int,
             pipelines=None) -> designMatrix.MultimodalDesignMatrix:
        """ Draw samples corresponding to the provided index """
        pipelines = self.__filterPipelines(pipelines)
        toLoad = self._loader.getIndexOfAllFromClass(classIndex)
        designMatrix = self.__loadDesignMatrixFromPipelines(
            sampleIDs=toLoad,
            fromPipelines=pipelines)
        return designMatrix

    def loadAllFromPipeline(self,
            pipeline: int) -> designMatrix.UnimodalDesignMatrix:
        """ Draw All samples from a chosen mode """
        toLoad = np.arange(self._loader.getSize())
        multiMatrix = self.__loadDesignMatrixFromPipelines(
            sampleIDs=toLoad,
            fromPipelines=[pipeline,])
        singleMatrix = multiMatrix.getModeByIndex(pipeline)
        return singleMatrix

    # Public Interface - Draw Samples

    def drawNext(self,
             sampleCount: np.ndarray,
             pipelines=None) -> designMatrix.MultimodalDesignMatrix:
        """ Draw samples corresponding to the provided index """
        pipelines = self.__filterPipelines(pipelines)
        toLoad = self._loader.getIndexOfNextOrdered(sampleCount)
        designMatrix = self.__loadDesignMatrixFromPipelines(
            sampleIDs=toLoad,
            fromPipelines=pipelines)
        return designMatrix

    def drawNextRandom(self,
             sampleCount: np.ndarray,
             pipelines=None) -> designMatrix.MultimodalDesignMatrix:
        """ Draw samples corresponding to the provided index """
        pipelines = self.__filterPipelines(pipelines)
        toLoad = self._loader.getIndexOfNextRandom(sampleCount)
        designMatrix = self.__loadDesignMatrixFromPipelines(
            sampleIDs=toLoad,
            fromPipelines=pipelines)
        return designMatrix

    # Public Interface - Operations

    def modeReport(self,toConsole=True) -> str:
        """ Return a report on the modes/pipelines in this dataset """
        txt = "\n" + "="*64 + "\n"
        for ii in range(len(self._pipelines)):
            pipelineInfo = "\n" + self._pipelines[ii].report()
            txt += pipelineInfo
        txt = "\n" + "="*64 + "\n"
        if (toConsole == True):
            print(txt)
        return txt
 
    def resetOrderedDraws(self) -> None:
        """ Reset & shuffle the list of drawn samples """
        self._loader.resetOrderedIter()
        return None

    def resetRandomDraws(self) -> None:
        """ Reset & shuffle the list of drawn samples """
        self._loader.resetRandomIter()
        return None

    def shuffleRandom(self) -> None:
        """ (re)-Shuffle and reset random draws """
        self._loader.shuffle()

    def logMessage(self, message: str) -> None:
        """ Log a message to the console """
        if (self._verbose == True):
            print(message)
        return None

    # Private Interface

    def __logMessage(self, message: str) -> None:
        """ Log a message to the console """
        if (self._verbose == True):
            print(message)
        return None

    def __validateRootPath(self) -> None:
        """ Validate that the root path exists """
        if (os.path.isdir(self._rootPath) == False):
            msg = "Provided root path at {0} does not exist".format(self._rootPath)
            raise RuntimeError(msg)
        return None

    def __filterPipelines(self,pipelineList: list) -> list:
        """ Given a list of ints, names, or both, return a list of ints indicating what pipelines to load """
        if (pipelineList is None):
            # If no list provided, load all pipelines
            return list(range(len(self._pipelines)))
        setOfPipelines = set()
        names = self.getPipelineNames()
        strToInt = lambda name : next((ii for ii, x in enumerate(names) if x == name),None)
        for item in pipelineList:
            if (isinstance(item,int) == True):
                if ((item >= 0) and (item < len(self._pipelines))):
                    setOfPipelines.add(item)
            elif (isinstance(item,str) == True):
                idx = strToInt(item)
                if (idx is not None):
                    setOfPipelines.add(idx)
        return list(setOfPipelines)


    def __loadClassInfo(self) -> None:
        """ Load in the class info database """
        classInfoFile = os.path.join(self._rootPath,"classInfo.txt")
        self._classInfo.readFromFile(classInfoFile)
        totals = self._classInfo.getTotals()
        totalNumSampleExported = totals[2]
        self._loader.resizeDatabase(totalNumSampleExported)
        return None

    def __loadPipelines(self) -> None:
        """ Load in each pipeline dataset as a tf dataset """
        rootContents = os.listdir(self._rootPath)
        MODE = "mode"
        for item in rootContents:
            fullRoot = os.path.join(self._rootPath,item)
            if (os.path.isdir(fullRoot) == False):
                continue
            if (item.startswith(MODE) == False):
                continue          
            self.__logMessage("Adding {0} to list of pipelines".format(item))
            pipelineCounter = len(self._pipelines)
            pipeline = datasetPipeline.DatasetPipeline(fullRoot,self,pipelineCounter)
            self._pipelines.append(pipeline)
        return None

    def __loadDesignMatrixFromPipelines(self,
               sampleIDs: np.ndarray,
               fromPipelines: list) -> designMatrix.MultimodalDesignMatrix:
        """ Load the chosen sample indices into a design matrix """
        timeStart = datetime.datetime.now()
        matrix = designMatrix.MultimodalDesignMatrix(
            numSamples=sampleIDs.size,
            numModes=self.getNumPipelines(),
            modeNames=self.getPipelineNames())
        # Populate design matrix
        matrix.setLabels( self._loader.getTargets(sampleIDs) )
        for pipelineID in fromPipelines:
            self.__loadFeaturesFromPipeline(matrix,sampleIDs,pipelineID)
        # Log Completion Message
        timeFinished = datetime.datetime.now()
        timeElapsed = timeFinished - timeStart
        msg = "Loaded {0} sample(s) from {1} pipeline(s). Time Elapsed: {2}".format(
            sampleIDs.size,len(fromPipelines),timeElapsed)
        self.__logMessage(msg)
        return matrix

    def __loadFeaturesFromPipeline(self,
                                    matrix: designMatrix.MultimodalDesignMatrix,
                                    samples: np.ndarray,
                                    pipelineID: int) -> None:
        """ Load in all features for the chosen samples for the chosen pipeline """
        labels = matrix.getLabels()
        features = self._loader.load(samples,labels,pipelineID)
        features = self.__applyReshapeToFeaturesWherePossible(
            features,pipelineID)
        matrix.setFeatures( pipelineID, features)
        return None

    def __applyReshapeToFeaturesWherePossible(self,
                                              features: np.ndarray,
                                              pipelineID: int) -> np.ndarray:
        """ Attempt to reshape features where possible """
        listOfShapes = self._pipelines[pipelineID].getShapes()
        # For NOW: We only reshape if there is a SINGLE feature type
        if (len(listOfShapes) == 1):
            # If only 1 feature type - reshape it
            numSamples = features.shape[0]
            newShape = (numSamples,) + listOfShapes[0]
            features = features.reshape(newShape)
        return features
