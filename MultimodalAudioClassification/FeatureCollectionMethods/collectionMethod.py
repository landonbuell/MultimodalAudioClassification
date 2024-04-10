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
        self._data  = np.zeros(shape=(numFeatures,),dtype=np.float32)
        self._callbacks = []    # called in __evaluateCallbacks() 
  
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getName(self) -> str:
        """ Return the name of this feature method """
        return self._name

    def getFeatures(self) -> np.ndarray:
        """ Return the populated features """
        return self._data

    def getNumFeatures(self) -> int:
        """ Return the number of features returned by this method """
        return self._data.size

    # Public Interface

    def call(self,
             signal: signalData.SignalData) -> np.ndarray:
        """ Evaluate this method on the provided signal """
        if (self.__evaluateCallbacks(signal) == False):
            return False
        if (self._callBody(signal) == False):
            return False
        if (self.__dataHasNanAndInfs() == True):
            return False
        return True

    def clear(self) -> None:
        """ Zero all features """
        for ii in range(self._data.size):
            self._data[ii] = 0.0
        return None

    def featureNames(self) -> list:
        """ VIRTUAL: Return a list of the feature names """
        result = ["{0}{1}".format(self._name,x) for x in range(self._data.size)]
        return result

    # Protected Interface 
    
    def _validateNumFeatures(self,numFeatures: int) -> int:
        """ VIRTUAL: Validate the number of output features """
        if (numFeatures <= 0):
            msg = "Error: Cannot have less than 1 output feature per collection method. Defaulting to 1 feature"
            print(msg)
            numFeatures = 1
        return numFeatures

    def _callBody(self,
                  signal: signalData.SignalData) -> bool:
        """ VIRTUAL: main body of call function """
        return False

    # Private Interface

    def __evaluateCallbacks(self,
                            signal: signalData.SignalData) -> bool:
        """ Evaulate callbacks on input signal. Return T/F if we can procceed """
        results = [True] * len(self._callbacks)  # Will be emty if len() == 0
        for ii,item in enumerate(self._callbacks):
            results[ii] = item.__call__(signal)
        numPasses = sum(results)
        return (numPasses == len(self._callbacks))

    def __dataHasNanAndInfs(self) -> bool:
        """ Check for NaN or +/- Inf Values """
        sumOfData = np.sum(self._data)
        if (sumOfData == np.inf) or (sumOfData == -np.inf):
            return True
        if (sumOfData == np.nan):
            return True
        return False


