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
import pandas as pd

    #### CLASS DEFINITIONS ####

class UnimodalDesignMatrix:
    """ Stores labels + corresponding features for samples """

    def __init__(self,
                 numSamples: int,
                 modeName=""):
        """ Constructor """
        self._labels    = np.ones(shape=(numSamples,),dtype=int) * -1
        self._features  = np.zeros(shape=(numSamples,1),dtype=np.float32)
        self._featureNames  = []
        self._featureShapes = []
        self._modeName  = modeName

    def __del__(self):
        """ Destructor """
        pass

    # ACCESSORS

    def numSamples(self) -> int:
        """ Number of samples present """
        return self._labels.size

    def numFeatures(self) -> int:
        """ Return the number of features """
        return int(self._featureNames.size / self._labels.size)

    def getLabels(self) -> np.ndarray:
        """ Return the labels """
        return self._labels

    def setLabels(self, labels: np.ndarray) -> None:
        """ Set the labels """
        self._labels = labels
        return None

    def getFeatures(self) -> np.ndarray:
        """ Get the features from a pipeline """
        return self._features

    def setFeatures(self, features: np.ndarray) -> None:
        """ Set the features for a pipeline """
        self._features = features
        return None

    def getFeatureNames(self) -> list:
        """ Return the list of feature Names """
        if (len(self._featureNames) == self.numFeatures):
            return self._featureNames
        return ["p{0}".format(x) for x  in range(len(self.numFeatures))]

    def setFeatureNames(self, names: list) -> None:
        """ Set the list of feature Names """
        if (len(names) == self.numFeatures):
            self._featureNames = names[:]
        return None

    def getShape(self) -> tuple:
        """ Return the shape of the features at the chosen index """
        return self._features.shape

    def getName(self) -> str:
        """ Return the name of the matrix """
        return self._modeName

    def setName(self, name: str) -> None:
        """ Set the name of the matrix """
        self._modeName = name
        return None

    # PUBLIC INTERFACE 

    def toDataFrame(self) -> pd.DataFrame:
        """ Cast multimodal design matrix to a list of Pandas DataFrames """
        frame = pd.DataFrame(data={"targets":self._labels},)
        return frame

    def getSample(self, sampleIndex: int) -> np.ndarray:
        """ Return a sample at the provided index """
        return self._features[sampleIndex]

    def getLabel(self, labelIndex: int) -> int:
        """ Return a sample at the provided index """
        return self._labels[labelIndex]

    # MAGIC METHODS

    def __len__(self) -> int:
        """ Number of samples present """
        return self._features.size

    def __repr__(self) -> str:
        """ Debug Representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))

class MultimodalDesignMatrix:
    """ Stores labels + corresponding features for samples """

    def __init__(self,
                 numSamples: int,
                 numModes: int,
                 modeNames=None):
        """ Constructor """
        self._labels    = np.ones(shape=(numSamples,),dtype=int) * -1
        self._features  = [None] * numModes
        self._modeNames = []

        if (len(modeNames) == numModes):
            self._modeNames = [str(x) for x in modeNames]
        else:
            msg = "Number of mode names does not match the number of modes"
            return RuntimeError(msg)

    def __del__(self):
        """ Destructor """
        pass

    # ACCESSORS

    def numSamples(self) -> int:
        """ Number of samples present """
        return self._labels.size

    def numModes(self) -> int:
        """ Number of modes """
        return len(self._features)

    def getModeNames(self) -> list:
        """ Return the list of mode names """
        return self._modeNames[:]

    def allFeatures(self) -> list:
        """ Return ALL features """
        self._features

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

    def toModes(self) -> list:
        """ Return a List of Unimodal Design Matrices """
        matrices = [None] * len(self._features)
        for ii in range(len(matrices)):
            matrices[ii] = UnimodalDesignMatrix(self.numSamples())
            matrices[ii].setName(       self._modeNames[ii])
            matrices[ii].setFeatures(   self._features[ii] )
            matrices[ii].setLabels(     np.copy(self._labels) )
        return matrices

    def getModeByName(self,modeName: str) -> UnimodalDesignMatrix:
        """ Get a Unimodal Design Matrix by name id """
        modeIndex = -1
        for ii,name in enumerate(self._modeNames):
            if (name == modeName):
                # names match
                modeIndex = ii
                break
        # Load using the mode ID
        if (modeIndex == -1):
            msg = "Cannot match name '{0} to a mode index".format(modeName)
            raise RuntimeError(msg)
        return self.getModeByIndex(modeIndex)

    def getModeByIndex(self,modeIndex: int) -> UnimodalDesignMatrix:
        """ Get a Unimodal Design Matrix by integer id """
        if (modeIndex < 0) or (modeIndex >= self.numModes()):
            msg = "Invalid mode index"
            raise RuntimeError(msg)

        matrix = UnimodalDesignMatrix(
            numSamples=self.numSamples(),
            modeName=self._modeNames[modeIndex])
        matrix.setFeatures(self._features[modeIndex])
        matrix.setLabels(np.copy(self._labels))
        return matrix

    # MAGIC METHODS

    def __len__(self) -> int:
        """ Number of samples present """
        return self._features.size

    def __repr__(self) -> str:
        """ Debug Representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))


