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
import queue
import numpy as np

import sessionInformation

        #### CLASS DEFINITIONS ####

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
        self._seed          = seed
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

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getRoot(self) -> str:
        """ Return the root of the dataset """
        return self._root

    def getSeed(self) -> int:
        """ Return the random seed for drawing samples """
        return self._seed

    def setSeed(self, newSeed: int) -> None:
        """ Set the random seed for this dataset """
        self._seed = newSeed
        return None

    def getClassInfo(self, classIndex: int) -> object:
        """ Return info about a class """
        return self._classInfo[classIndex]

    # Public Interface

    def draw(self,numSamples=1) -> object:
        """ Draw a single random X,y pair from the dataset """
        # TODO: THIS!
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
            self._classes       = list()
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
            self._database      = np.ones(shape=(1024,),dtype=int) * -1

            

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

        def populate(self):
            """ Populate this database """
            sampleFiles = self.__findSampleFiles()
            self.__countAllSamples(sampleFiles)
            return None
        
        def resizeDatabase(self, newSize: int) -> None:
            """ Resize the internal database """
            sizeDiff = newSize - self._database.size
            if (sizeDiff >= 1):
                extension = np.ones(shape=(sizeDiff,),dtype=int) * -1
                self._database.append(extension)
            return None

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

        def __countAllSamples(self, sampleFiles: list) -> None:
            """ Count all samples between all sample files """
            rootContents = os.listdir(self._rootPath)
            DOT_TXT = ".txt"
            SAMPLES = "samples"
            sampleFiles = list()
            for item in rootContents:
                fullPathToItem = os.path.join(self._rootPath,item)
                if (os.path.isfile(fullPathToItem) == False):
                    continue
                if (item.endswith(DOT_TXT) == False):
                    continue
                if (item.startswith(SAMPLES) == False):
                    continue
                self.__countSamplesInFile(fullPathToItem)
            return None

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

