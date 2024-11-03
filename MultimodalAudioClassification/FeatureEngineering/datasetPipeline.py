"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureEngineering
    File:       datasetPipeline.py
    Classes:    DatasetPipeline

    Author:     Landon Buell
    Date:       April 2024
"""

    #### IMPORTS ####

import os
from typing_extensions import runtime
import numpy as np

import designMatrix

    #### CLASS DEFINITIONS ####

class DatasetPipeline:
    """ Stores all information related to a pipeline from within a dataset """

    def __init__(self,
                 pipelineName: str,
                 parentDataset: object,
                 identifier: int):
        """ Constructor """
        self._pipelineName  = pipelineName
        self._parentDataset = parentDataset
        self._identifier    = identifier
        self._numFeatures   = 0
        self._classes       = dict() # int -> str
        self._shapes        = list()
        self._names         = list()
        self._getPath       = lambda x,y : os.path.join(self._rootPath,"class{0}".format(x),"sample{0}.bin".format(y))
            
        self.__loadShapes()


    def __del__(self):
        """ Destructor """
        self._classes.clear()
        self._shapes.clear()
        self._names.clear()

    # Accessors

    def getRoot(self) -> str:
        """ Return the root path of the pipeline """
        return os.path.join(self._parentDataset.getRoot(),self._pipelineName)

    def getNumFeatures(self) -> int:
        """ Number of features in this pipeline """
        return self._numFeatures

    def getClasses(self) -> list:
        """ Return a list of the classes processed by this pipeline """
        return self._classes

    def getTargetPath(self, targetID: int) -> str:
        """ Return the path of the labels folder """
        return os.path.join( self.getRoot(), "class{0}".format(targetID))

    def getSamplePath(self,sampleID: int, targetID: int) -> str:
        """ Return the path of the chosen sample given the target label """
        return os.path.join(self.getTargetPath(targetID),"sample{0}.bin".format(sampleID))

    # Public Interface
    
    # Private Interface

    def __logMessage(self, message: str) -> None:
        """ Log message via the parent dataset """
        self._parentDataset.logMessage(message)
        return None

    def __loadShapes(self) -> None:
        """ Load in all of the shapes """
        shapesFile = os.path.join(self.getRoot(),"featureShapes.txt")
        with open(shapesFile,"r") as inputStream:
            for ii,line in enumerate(inputStream):
                if (ii == 0):
                    # Skip header row
                    continue
                lineTokens = line.strip().split()
                self._numFeatures += int(lineTokens[1])
                shapeTuple = tuple(lineTokens[2].split(","))
                self._shapes.append(shapeTuple)
        return None
