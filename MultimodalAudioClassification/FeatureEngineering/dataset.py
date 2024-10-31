"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureEngineering
    File:       dataset.py
    Classes:    DatasetLoader

    Author:     Landon Buell
    Date:       April 2024
"""

        #### IMPORTS ####

import os
import threading

import numpy as np

import sessionInformation

        #### CLASS DEFINITIONS ####

class LabelFeaturePair:
    """ Stores a pair of features & corresponding labels """

    def __init__(self,
                 labels: np.ndarray):
        """ Constructor """
        self.labels     = labels
        self.features   = [None] * labels.size

    def __del__(self):
        """ Destructor """
        pass

    def __len__(self):
        """ Number of samples present """
        self.labels.size


class Dataset:
    """ 
        Represents a dataset that can be loaded for later use 
    """

    SAMPLES_PER_FILE = 1000     # Fetch from RunInfo Later

    def __init__(self,
                 rootPath: str,
                 seed=123456789,
                 verbose=True):
        """ Constructor """
        self._rootPath      = rootPath
        self._verbose       = verbose
        self.__validateRootPath()

        # Dataset Members
        self._runInfo       = sessionInformation.RunInfo()
        self._classInfo     = sessionInformation.ClassInfoDatabase()
        self._sampleDatabase = Dataset.__SampleDatabase(rootPath,self)
        self._pipelines     = []

        # Load in all of the internal information
        self.__loadClassInfo()
        self.__loadPipelines()
        self._sampleDatabase.populate()
        self._sampleDatabase.shuffle()

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getRoot(self) -> str:
        """ Return the root of the dataset """
        return self._root

    def getClassInfo(self, classIndex: int) -> object:
        """ Return info about a class """
        return self._classInfo[classIndex]

    # Public Interface

    def draw(self,numSamples=1,fromPipelines=None) -> object:
        """ Draw a single random X,y pair from the dataset """
        toDraw = self._sampleDatabase.getNext(numSamples)
        result = LabelFeaturePair(labels=toDraw[0])
        if (fromPipelines is None):
            fromPipelines = list(range(len(self._pipelines)))
        for 

        return None

    def resetDraws(self) -> None:
        """ Reset & shuffle the list of drawn samples """
        return None

    def sourceOfSample(self, index: int) -> str:
        """ Return the source path for the provided sample """
        return ""

    # Public Nested Class

    class Pipeline:
        """ Stores all information related to a pipeline """

        def __init__(self,
                     rootPath: str,
                     parentDataset: object):
            """ Constructor """
            self._rootPath      = rootPath
            self._parentDataset = parentDataset
            self._loader        = 
            self._classes       = dict() # int -> str
            self._shapes        = list()
            self._names         = list()

        def __del__(self):
            """ Destructor """
            self._classes.clear()
            self._shapes.clear()
            self._names.clear()

        # Accessors

        def getClasses(self) -> list:
            """ Return a list of the classes processed by this pipeline """
            return self._classes

        # Public Interface

        def loadSamplesFromClass(self,
                        sampleIdentifiers: np.ndarray,
                        targetClasses: np.ndarray):
            """ Load the sample corresponding to the identifiers """
            
            return None



        # Private Interface

    # Private Nested Classes

    class __SampleDatabase:
        """ Stores a database of all known samples """

        DEFAULT_SIZE = int(2**10)

        def __init__(self,
                     rootPath: str,
                     parentDataset: object):
            """ Constructor """
            self._rootPath      = rootPath
            self._parentDataset = parentDataset
            self._actualSize    = 0
            self._database      = np.ones(shape=(256,),dtype=int) * -1
            self._shuffled      = np.zeros(shape=self._database.shape)
            self._shuffledIter  = 0

        def __del__(self):
            """ Destructor """
            pass

        # Accessors

        def getSize(self) -> int:
            """ Return the actual number of samples """
            return self._actualSize

        def __getitem__(self,key: int):
            """ Index an item in the database """
            return self._database[key]

        def __setitem__(self,key: int, val: int):
            """ Set an item in the database """
            self._database[key] = val
            return None

        # Public Interface

        def populate(self) -> None:
            """ Populate this database """
            sampleFiles = self.__findSampleFiles()
            for item in sampleFiles:
                self.__countSamplesInFile(os.path.join(self._rootPath,item))
            self.shuffle()
            return None

        def shuffle(self) -> None:
            """ Shuffles the order that samples will be drawn. Resets the internal iterator """
            self._shuffled = np.arange(self._database.size)
            np.random.shuffle(self._shuffled)
            self._shuffledIter = 0
            return None

        def resizeDatabase(self, newSize: int) -> None:
            """ Resize the internal database, reset random iterator """
            sizeDiff = newSize - self._database.size
            if (sizeDiff >= 1):
                extension = np.ones(shape=(sizeDiff,),dtype=int) * -1
                self._database = np.append(self._database,extension)
                self._shuffled = np.zeros(shape=self._database.shape)
                self._shuffledIter = 0
            return None

        def getNext(self,count=1) -> list:
            """ Draw the next 'count' number of samples """            
            upperBound = np.min([self._shuffledIter + count,self._database.size])
            actualCount = upperBound - self._shuffledIter
            result = np.ones(shape=(2,actualCount),dtype=int) * -1
            result[0] = self._shuffled[self._shuffledIter:upperBound] # samples to draw
            result[1] = self._database[self._shuffledIter:upperBound] # corresponding classes
            self._shuffledIter = (self._shuffledIter + actualCount) % self._database.size
            return result


        # Private Interface

        def __findSampleFiles(self) -> list:
            """ Load all of the sample files in this data set """
            rootContents = os.listdir(self._rootPath)
            DOT_TXT = ".txt"
            SAMPLES = "samples"
            sampleFiles = list()
            for item in rootContents:
                if (os.path.isfile(os.path.join(self._rootPath,item)) == False):
                    continue
                if (item.endswith(DOT_TXT) == False):
                    continue
                if (item.startswith(SAMPLES) == False):
                    continue
                sampleFiles.append(item)
            return sampleFiles

        def __countSamplesInFile(self,filePath) -> int:
            """ Read the provided file and count the number of samples in it """
            with open(filePath,"r") as inputStream:
                for ii,line in enumerate(inputStream):
                    if (ii == 0):
                        # Always skip the header
                        continue
                    lineTokens = line.strip().split()
                    sampleIndex = int(lineTokens[0])
                    classIndex = int(lineTokens[1])
                    if (sampleIndex >= self._database.size):
                        self.resizeDatabase(np.max([sampleIndex,self._database.size * 2]))
                    if (self._database[sampleIndex] == -1):
                        self._actualSize += 1
                    self._database[sampleIndex] = classIndex
            return None

    # Private Interface

    def __logMessage(self, message: str) -> None:
        """ Log a message to the console """
        if (self._verbose == True):
            print(message)
        return None

    def __validateRootPath(self) -> None:
        """ Validate that the root path exists """
        if (os.path.isdir(self._rootPath) == False):
            msg = "Provided root path at {0} does not exist".format(self._rootPath)
            raise RuntimeError(msg)
        return None

    def __loadClassInfo(self) -> None:
        """ Load in the class info database """
        classInfoFile = os.path.join(self._rootPath,"classInfo.txt")
        self._classInfo.readFromFile(classInfoFile)
        totals = self._classInfo.getTotals()
        totalNumSampleExported = totals[2]
        self._sampleDatabase.resizeDatabase(totalNumSampleExported)
        return None

    def __loadPipelines(self) -> None:
        """ Load in each pipeline dataset as a tf dataset """
        rootContents = os.listdir(self._rootPath)
        PIPELINE = "pipeline"
        for item in rootContents:
            fullRoot = os.path.join(self._rootPath,item)
            if (os.path.isdir(fullRoot) == False):
                continue
            if (item.startswith(PIPELINE) == False):
                continue
            self.__logMessage("Adding {0} to list of pipelines".format(item))
            pipeline = Dataset.Pipeline(fullRoot,self)
            self._pipelines.append(pipeline)
        return None

