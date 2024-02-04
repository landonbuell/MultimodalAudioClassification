"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       featureVector.py
    Classes:    FeatureVector 

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import numpy as np

        #### CLASS DEFINITIONS ####

class FeatureVector:
    """ Stores a 1D Vector of features and a corresponding target class label """

    def __init__(self,
                 numFeatures: int,
                 classLabel: int):
        """ Constructor """
        self._data = np.zeros(shape=(numFeatures,),dtype=np.float32)
        self._target = classLabel

    def __del__(self):
        """ Destructor """
        pass

    # Accessors 

    def getNumFeatures(self) -> int:
        """ Return the number of features """
        return self._data.size

    def getTarget(self) -> int:
        """ Return the target class """
        return self._target

    def setTarget(self,
                  classLabel: int) -> None:
        """ Set the target value """
        self._target = classLabel
        return None

    # Public Interface

    def clear(self) -> None:
        """ Reset all entries to zeros """
        numFeatures = self._data.size
        self._data = np.zeros(shape=(numFeatures,),dtype=np.float32)

    def toBinaryFile(self,
               fullOutputPath: str) -> bool:
        """ Export the contents of this class to a file """
        success = False
        try:
            self._data.tofile(fullOutputPath)
            success = True
        except Exception as err:
            msg = "Failed to write bytes to {0} for reason: {1}".format(fullOutputPath,str(err))
            raise RuntimeWarning(msg)
        return success

    def toTextFile(self,
               fullOutputPath: str) -> bool:
        """ Export the contents of this class to a file """
        success = False
        try:
            self._data.tofile(fullOutputPath,sep=" ")
            success = True
        except Exception as err:
            msg = "Failed to write data to {0} for reason: {1}".format(fullOutputPath,str(err))
            raise RuntimeWarning(msg)
        return success

    # Private Interface
   
