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
import sys

import numpy as np

import PyToolsIO


        #### CLASS DEFINITIONS ####

class FeatureVector:
    """ Class to Hold Feature Data for a single Sample """

    def __init__(self,sampleShape,label=-1):
        """ Constructor for FeatureVector Instance """
        self._label         = label
        self._data          = np.zeros(shape=sampleShape,dtype=np.float32)

    def __del__(self):
        """ Destructor for FeatureVector Instance """
        self.clearData()

    # Getters and Setters

    def getShape(self):
        """ Get the Shape of this Sample """
        return self._data.shape

    def getLabel(self):
        """ Get the Target Label """
        return self._label

    def setLabel(self,x):
        """ Set the Target Label """
        self._label = x
        return self

    def getData(self):
        """ Get the underlying Array """
        return self._data

    def setData(self,x,enforceShape=True):
        """ Set the Underlying Array, optionally chanign shape """
        if (enforceShape == True):
            assert(list(x.shape) == self.getShape())
            self._data = x
        else:
            self._sampleShape = x.shape
            self._data = x
        return self

    # Public Interface

    def clearData(self):
        """ Clear All Entries in this Array """
        dataShape           = self._data.shape
        self._label         = -1
        self._data          = np.zeros(shape=dataShape,dtype=np.float32)
        return self

    # Magic Method

    def __str__(self):
        """ String Representation of Instance """
        return str(self.__class__) + " w/ shape: " + str(self.getShape())

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

    def __iter__(self):
        """ Forward-Iterator through Design Matrix """
        for i in range(self._data.shape[0]):
            yield self._data[i]

    def __getitem___(self,key):
        """ Get the Item at the Index """
        return self._data[key]

    def __setitem__(self,key,value):
        """ Set the Item at the Index """
        self._data[key] = np.float32(value)
        return self

class DesignMatrix:
    """ Class To hold Design Matrix """

    def __init__(self,numSamples: int,sampleShape: tuple):
        """ Constructor for DesignMatrix Instance """
        self._numSamples    = numSamples 
        self._sampleShape   = sampleShape
        self._data          = np.zeros(shape=self.getMatrixShape(),dtype=np.float32)
        self._tgts          = np.zeros(shape=numSamples,dtype=np.int16)

    def __del__(self):
        """ Destructor for DesignMatrix Instance """
        self.clearData()

    # Getters and Setters

    def getMatrixShape(self):
        """ Get Total Shape of Design Matrix """
        shape = [self._numSamples] + [x for x in self._sampleShape]
        return tuple(shape)

    def getSampleShape(self):
        """ Get the Shape of Each Sample in the Design Matrix """
        return self._sampleShape

    def getNumFeatures(self):
        """ Get the Total Number of Features for each sample """
        numFeatures = 1
        for axisSize in self._sampleShape:
            numFeatures *= axisSize
        return numFeatures

    def getNumSamples(self) -> int:
        """ Get the Number of Samples in the Design Matrix """
        return self._numSamples

    def setNumSamples(self,numSamples):
        """ Set the Number of Samples in the Design Matrix """
        self._numSamples = numSamples
        self.clearData()
        return self

    def getFeatures(self):
        """ Get Design Matrix as an Array """
        return self._data

    def setFeatures(self,x):
        """ Set Design Matrix is an Array """
        self._numSamples = x.shape[0]
        self._sampleShape = tuple(x.shape[1:])
        self._data = None
        self._data = x
        return self

    def getLabels(self):
        """ Get the Labels as an Array """
        return self._tgts

    def setLabels(self,x):
        """ Set the Labels as an Array """
        self._tgts = x
        return self

    def getUniqueClasses(self):
        """ Get An Array of the unique classes """
        return np.unique(self._tgts)

    def getNumClasses(self):
        """ Get the Number of classes in the data set """
        return np.unique(self._tgts).shape[-1]

    # public Interface

    def serialize(self,pathX,pathY):
        """ Write this design matrix out to a file """   
        writerX = PyToolsIO.DesignMatrixDataSerializer(self,pathX)
        writerY = PyToolsIO.DesignMatrixLabelSerializer(self,pathY)
        success = True
        try:          
            success = writerX.call()
            success = writerY.call()
        except Exception as err:
            print("\t\tDesignMatrix.serialize()" + err)
            success = False
        return success

    @staticmethod
    def deserialize(pathX,pathY,numSamples,shape):
        """ Read a design matrix from a file """
        msg = "{0}.deserialize is not implement".format("DesignMatrix")
        raise RuntimeError(msg)

    def clearData(self):
        """ Clear All Entries in this Array """
        self._data = np.zeros(shape=self.getShape(),dtype=np.float32)
        self._tgts = np.zeros(shape=self.getNumSamples(),dtype=np.int16)
        return self

    @staticmethod
    def encodeOneHot(targets,numClasses):
        """ Get a One-Hot-Encoded Array of targets """
        numSamples = targets.shape[-1]
        result = np.zeros(shape=(numSamples,numClasses),dtype=np.int16)   
        for i in range(numSamples):
            tgt = targets[i]
            result[i,tgt] = 1
        return result

    # Private Interface
 
    # Magic Methods 

    def __str__(self):
        """ String Representation of Instance """
        return str(self.__class__) + " w/ shape: " + str(self.getShape())

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

    def __iter__(self):
        """ Forward-Iterator through Design Matrix """
        for i in range(self._data.shape[0]):
            yield self._data[i]

    def __getitem__(self,key):
        """ Get the Item at the Index """
        if (key < 0 or key >= self._numSamples):
            errMsg = "key index is out of range for " + self.__repr__
            raise IndexError(errMsg)
        # Make a Feature Vector + Return it
        featureVector = FeatureVector(self._sampleShape,self._tgts[key])
        featureVector.setData(self._data[key])
        return featureVector

    def __setitem__(self,key,value):
        """ Set the Item at the Index """
        assert(value.getShape() == self._sampleShape)
        self._tgts[key] = value.getLabel()
        self._data[key] = value.getData()
        return self

class RunInfo:
    """
    Class to Hold and Use all Metadata related to a feature collection Run
    """

    def __init__(self,inputPaths,outputPath):
        """ Constructor for RunInformation Instance """
        self._pathsInput        = inputPaths
        self._pathOutput        = outputPath

        self._batchSizes        = []
        self._pipelineInfo      = []


    def __del__(self):
        """ Destructor for RunInformation Instance """
        pass

    # Getters and Setters

    def getRunInfoPath(self):
        """ Get the Path to the RunInfo Metadata """
        return os.path.join(self._pathOutput,"runInfo.txt")

    def getInputPaths(self) -> set:
        """ Return List of Input Paths """
        return self._pathsInput

    def getOutputPath(self) -> str:
        """ Return Output Path """
        return self._pathOutput

    def getBatchSizes(self):
        """ Get a list with the size of each batch """
        return self._batchSizes

    def getNumBatches(self):
        """ Get the Number of Batches in the run """
        return len(self._batchSizes)

    def addBatchSize(self,batchSize):
        """ Append to the list of batch sizes """
        self._batchSizes.append(batchSize)
        return self

    def getNumSamplesProcessed(self):
        """ Get the Totoal Number of Samples Seen """
        return sum(self._batchSizes)
    
    # Public Interface 

    def serialize(self,path):
        """ Serialize this Instance to specified Path """
        writer = PyToolsIO.RunInfoSerializer(self,path)
        success = False
        try:
            success = writer.call()
        except Exception as expt:
            print(expt)
        return success

    @staticmethod
    def deserialize(path):
        """ Deserialize this instance from specified path """
        msg = "{0}.deserialize is not implement".format("RunInfo")
        raise RuntimeError(msg)

    # Private Interface
        
    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))
