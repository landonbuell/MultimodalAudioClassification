"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       collectionMethod.py
    Classes:    AbstractCollectionMethod,
                CollectionMethodCallbacks

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import numpy as np
import enum

import signalData
import featureVector

        #### CLASS DEFINITIONS ####

class WeightingFunction(enum.IntEnum):
    """ Enumeration for Weighting Function Type """
    LINEAR      = 1
    QUADRATIC   = 2

    LOG_NATURAL = 9
    LOG_BASE10  = 10

class AbstractCollectionMethod:
    """ Abstract base class for all collection methods """

    DELTA = 1e-8

    def __init__(self,
                 methodName: str,
                 numFeatures: int):
        """ Constructor """
        numFeatures = self._validateNumFeatures(numFeatures)
        self._name  = methodName
        self._numFeatures = numFeatures
        self._shape = [numFeatures,]
        self._ptrPipeline  = None
        self._callbacks = []    # called in __evaluateCallbacks() 
  
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getName(self) -> str:
        """ Return the name of this feature method """
        return self._name

    def getNumFeatures(self) -> int:
        """ Return the number of features returned by this method """
        return self._numFeatures

    def getShape(self) -> list:
        """ Return the intended shape of the output features """
        return self._shape

    def _setIntendedShape(self,
                           shape: list) -> None:
        """ Set the intended shape of the output features """
        numFeaturesInShape = 1
        for axisSize in shape:
            numFeaturesInShape *= axisSize
        if (numFeaturesInShape != self._numFeatures):
            msg = "Cannot set intended shape of {0} w/ {1} features to {2} due to size mis-match.".format(
                self.getName(),self._numFeatures,str(shape))
            self._logMessage(msg)
            raise RuntimeError(msg)
        self._shape = shape
        return None

    def getPipeliene(self) -> object:
        """ Return the pipeline that owns this collection method """
        return self._ptrPipeline

    # Public Interface

    def call(self,
             signal: signalData.SignalData,
             features: featureVector.FeatureVector) -> np.ndarray:
        """ Evaluate this method on the provided signal """
        if (self.__evaluateCallbacks(signal) == False):
            return False
        if (self._callBody(signal,features) == False):
            return False
        return True

    def featureNames(self) -> list:
        """ VIRTUAL: Return a list of the feature names """
        numFeatures = self.getNumFeatures()
        result = ["{0}{1}".format(self._name,x) for x in range(numFeatures)]
        return result

    def registerPipeline(self,ptrPipeline) -> None:
        """ Register the pipeline that will run this collection method (opt) """
        self._ptrPipeline = ptrPipeline
        return None

    # Protected Interface 
    
    def _validateNumFeatures(self,numFeatures: int) -> int:
        """ VIRTUAL: Validate the number of output features """
        if (numFeatures <= 0):
            msg = "Error: Cannot have less than 1 output feature per collection method. Defaulting to 1 feature"
            print(msg)
            numFeatures = 1
        return numFeatures

    def _callBody(self,
                  signal: signalData.SignalData,
                  features: featureVector.FeatureVector) -> bool:
        """ VIRTUAL: main body of call function """
        return False

    def _logMessage(self,message) -> None:
        """ Log a message through the owning pipeline if possible """
        if (self._ptrPipeline is not None):
            ptrPipelineMgr = self._ptrPipeline.getManager()
            if (ptrPipelineMgr is not None):
                ptrPipelineMgr.logMessage(message)
        return None

    def _resizeData(self, newNumFeatures: int) -> None:
        """ Resize the internal data and shape to match a new size """
        self._numFeatures   = newNumFeatures
        self._shape         = [newNumFeatures,]
        return self

    # Private Interface

    def __evaluateCallbacks(self,
                            signal: signalData.SignalData) -> bool:
        """ Evaulate callbacks on input signal. Return T/F if we can procceed """
        results = [True] * len(self._callbacks)  # Will be emty if len() == 0
        for ii,item in enumerate(self._callbacks):
            results[ii] = item.__call__(signal)
        numPasses = sum(results)
        return (numPasses == len(self._callbacks))



