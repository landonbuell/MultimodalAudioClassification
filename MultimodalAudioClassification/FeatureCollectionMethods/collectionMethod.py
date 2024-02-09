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
             signal: signalData.SignalData) -> None:
        """ Evaluate this method on the provided signal """
        return None

    def clear(self) -> None:
        """ Zero all features """
        for ii in range(self._data.size):
            self._data[ii] = 0.0
        return None

    # Protected Interface 
