"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       collectionMethod.py
    Classes:    AbstractCollectionMethod

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import numpy as np

import signalData

        #### CLASS DEFINITIONS ####

class AbstractCollectionMethod:
    """ Abstract base class for all collection methods """

    def __init__(self,
                 methodName: str,
                 numFeatures: int):
        """ Constructor """
        self._name  = methodName
        self._data  = np.zeros(size=(numFeatures,),dtype=np.float32)
        self._callbacks = []    # evaluated 
        
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
             signal: signalData.SignalData) -> bool:
        """ Evaluate this method on the provided signal """
        if (self.__evaluateCallbacks(signal) == False):
            return False
        if (self._callBody(signal) == False):
            return False
        if (self.__dataHasNanAndInfs == True):
            return False
        return True

    def clear(self) -> None:
        """ Zero all features """
        for ii in range(self._data.size):
            self._data[ii] = 0.0
        return None

    # Protected Interface 

    def _callBody(self) -> bool:
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