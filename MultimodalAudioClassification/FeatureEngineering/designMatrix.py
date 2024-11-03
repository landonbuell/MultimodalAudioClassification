"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureEngineering
    File:       designMatrix.py
    Classes:    DesignMatrix

    Author:     Landon Buell
    Date:       November 2024
"""

    #### IMPORTS ####

import numpy as np

    #### CLASS DEFINITIONS ####

class MultimodalDesignMatrix:
    """ Stores labels + corresponding features for samples """

    def __init__(self,
                 numSamples: int,
                 numPipelines: int):
        """ Constructor """
        self._labels    = np.ones(shape=(numSamples,),dtype=int) * -1
        self._features  = [None] * numPipelines

    def __del__(self):
        """ Destructor """
        pass

    # ACCESSORS

    def numSamples(self) -> int:
        """ Number of samples present """
        return self._labels.size

    def allFeatures(self) -> list:
        """ Return ALL features """

    def getLabels(self) -> np.ndarray:
        """ Return the labels """
        return self._labels

    def setLabels(self, labels: np.ndarray) -> None:
        """ Set the labels """
        if (labels.size != self._labels.size):
            raise RuntimeError("Size mismatch when setting labels")
        self._labels = labels
        return None

    def getFeatures(self, pipelineIndex: int) -> np.ndarray:
        """ Get the features from a pipeline """
        return self._features[pipelineIndex]

    def setFeatures(self,pipelineIndex: int, features: np.ndarray) -> None:
        """ Set the features for a pipeline """
        if (features.shape[0] != self.numSamples()):
            raise RuntimeError("Size mismatch when setting features")
        self._features[pipelineIndex] = features
        return None

    def getShape(self, pipelineIndex: int) -> tuple:
        """ Return the shape of the features at the chosen index """
        if (self._features[pipelineIndex] is None):
            return (0,)
        return self._features[pipelineIndex].shape

    # PUBLIC INTERFACE 

    # MAGIC METHODS

    def __len__(self) -> int:
        """ Number of samples present """
        return self._features.size

    def __repr__(self) -> str:
        """ Debug Representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))


