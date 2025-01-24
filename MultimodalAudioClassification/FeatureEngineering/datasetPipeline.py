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
import numpy as np

import designMatrix

    #### CLASS DEFINITIONS ####

class DatasetPipeline:
    """ Stores all information related to a pipeline from within a dataset """

    def __init__(self,
                 rootPath: str,
                 parentDataset: object,
                 identifier: int):
        """ Constructor """
        self._rootPath      = rootPath
        self._parentDataset = parentDataset
        self._identifier    = identifier
        self._numFeatures   = 0
        self._classes       = dict() # int -> str
        self._shapes        = list()
        self._names         = list()
            
        self.__loadShapes()


    def __del__(self):
        """ Destructor """
        self._classes.clear()
        self._shapes.clear()
        self._names.clear()

    # Accessors

    def getRoot(self) -> str:
        """ Return the root path of the pipeline """
        return os.path.join(self._parentDataset.getRoot(),self._rootPath)

    def getNumFeatures(self) -> int:
        """ Number of features in this pipeline """
        return self._numFeatures

    def getClasses(self) -> list:
        """ Return a list of the classes processed by this pipeline """
        return self._classes.keys()

    def getTargetPath(self, targetID: int) -> str:
        """ Return the path of the labels folder """
        return os.path.join( self.getRoot(), "class{0}".format(targetID))

    def getSamplePath(self,sampleID: int, targetID: int) -> str:
        """ Return the path of the chosen sample given the target label """
        return os.path.join(self.getTargetPath(targetID),"sample{0}.bin".format(sampleID))

    def getName(self) -> str:
        """ Return the name of the pipeline (based on the root) """
        rootName = self._rootPath.split(os.pathsep)[-1]
        tokens = rootName.split("_")
        return tokens[-1]

    def getShapes(self) -> list:
        """ Return a list of tuples describing the shape of each group of features """
        return self._shapes[:]

    # Public Interface

    def report(self) -> str:
        """ Return a string that provides a detailed report on this instance """
        fmt = lambda n,x,y : "\t" * n + "{0:<32}{1}".format(x,y)
        txt = ""
        txt += fmt(1,"Pipleine #{0}".format(self._identifier),self._rootPath)
        txt += fmt(2,"Num Features",self._numFeatures)
        txt += fmt(2,"Num Classes", len(self._classes))
        return txt

    
    # Private Interface

    def __logMessage(self, message: str) -> None:
        """ Log message via the parent dataset """
        self._parentDataset.logMessage(message)
        return None

    def __getSampleFileRoot(self,classIndex: int, sampleIndex: int):
        """ Return the path to the unique sample """
        return os.path.join(self._rootPath,
                            "class{0}".format(classIndex),
                            "sample{0}.bin".format(sampleIndex))

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
                shapeTokens = lineTokens[2].split(".")
                shapeTuple = tuple([int(x) for x in shapeTokens])        
                self._shapes.append(shapeTuple)
        return None

    # Dunder Methods

    def __eq__(self,other) -> bool:
        """ Equality operator """
        if (isinstance(other,type(self) == False)):
            return False
        return ((self._rootPath == other._pipelineName) and
                (self._parentDataset == other._parentDataset) and
                (self._identifier == other._identifier))

    def __neq__(self,other) -> bool:
        """ In-Equality operator """
        return not (self == other)