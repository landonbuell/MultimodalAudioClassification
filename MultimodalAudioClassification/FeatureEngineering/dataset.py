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
import datetime

import numpy as np

import sessionInformation

        #### CLASS DEFINITIONS ####

class LabelFeaturePair:
    """ Stores a pair of features & corresponding labels """

    def __init__(self,
                 numSamples : int,
                 numPipelines: int):
        """ Constructor """
        self.labels     = np.ones(shape=(numSamples),dtype=int) * -1
        self.features   = [None] * numPipelines

    def __del__(self):
        """ Destructor """
        pass

    def __len__(self):
        """ Number of samples present """
        self.labels.size

    def __repr__(self) -> str:
        """ Debug Representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))


class Dataset:
    """ 
        Represents a dataset that can be loaded for later use 
    """

    SAMPLES_PER_FILE = 1000     # Fetch from RunInfo Later

    def __init__(self,
                 rootPath: str,
                 seed=123456789,
                 verbose=True,
                 multiThread=False):
        """ Constructor """
        self._rootPath      = rootPath
        self._verbose       = verbose
        self._multiThread   = multiThread
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
        self._sampleDatabase.shuffle(seed)

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getRoot(self) -> str:
        """ Return the root of the dataset """
        return self._root

    def getNumPipelines(self) -> int:
        """ Return the number of pipelines """
        return len(self._pipelines)

    def getNumFeatures(self,pipelineIndex: int) -> int:
        """ Return the number of features in the chosen pipeline """
        return self._pipelines[pipelineIndex].getNumFeatures()

    def getClassInfo(self, classIndex: int) -> object:
        """ Return info about a class """
        return self._classInfo[classIndex]

    # Public Interface

    def draw(self,numSamples=1,fromPipelines=None) -> LabelFeaturePair:
        """ Draw a single random X,y pair from the dataset """
        timeStart = datetime.datetime.now()
        toDraw = self._sampleDatabase.getNext(numSamples)
        
        # Determine which pipelines to load data from
        if (fromPipelines is None):
            fromPipelines = list(range(len(self._pipelines)))
        xyPair = LabelFeaturePair(toDraw.size,len(fromPipelines))
        xyPair.labels = self._sampleDatabase.getTargets(toDraw)

        # Load from each selected pipeline
        loaderThreads = [None] * len(fromPipelines)
        for ii,pipelineIndex in enumerate(fromPipelines):
            loaderThreads[ii] = threading.Thread(
                target=self._pipelines[pipelineIndex].loadSamples,
                args=(toDraw,xyPair,ii))
        
        # Start the load process
        if (self._multiThread == True):
            for loader in loaderThreads:
                loader.start()
            for loader in loaderThreads:
                loader.join()
        else:
            for loader in loaderThreads:
                loader.run()

        timeFinish = datetime.datetime.now()
        timeElapsed = timeFinish - timeStart
        msg = "Loaded {0} samples from {1} pipelines. Time elapsed: {2}".format(
            toDraw.size,len(fromPipelines),timeElapsed)
        self.__logMessage(msg)
        return xyPair

    def resetDraws(self) -> None:
        """ Reset & shuffle the list of drawn samples """
        self._sampleDatabase.shuffle()
        return None

    def loadAllSamplesFromClass(self,classIndex: int, fromPipelines=None) -> LabelFeaturePair:
        """ Load all samples from a provided class """
        timeStart = datetime.datetime.now()
        toDraw = self._sampleDatabase.

        # Determine which pipelines to load data from
        if (fromPipelines is None):
            fromPipelines = list(range(len(self._pipelines)))
        xyPair = LabelFeaturePair(toDraw.size,len(fromPipelines))
        xyPair.labels = self._sampleDatabase.getTargets(toDraw)

        timeFinish = datetime.datetime.now()
        timeElapsed = timeFinish - timeStart
        msg = "Loaded {0} samples from {1} pipelines. Time elapsed: {2}".format(
            toDraw.size,len(fromPipelines),timeElapsed)
        self.__logMessage(msg)
        return xyPair

    def logMessage(self, message: str) -> None:
        """ Log a message to the console """
        if (self._verbose == True):
            print(message)
        return None

    # Private Nested Classes

    class __Pipeline:
        """ Stores all information related to a pipeline """

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
            self._getPath       = lambda x,y : os.path.join(self._rootPath,"class{0}".format(x),"sample{0}.bin".format(y))
            
            self.__loadShapes()


        def __del__(self):
            """ Destructor """
            self._classes.clear()
            self._shapes.clear()
            self._names.clear()

        # Accessors

        def getNumFeatures(self) -> int:
            """ Number of features in this pipeline """
            return self._numFeatures

        def getClasses(self) -> list:
            """ Return a list of the classes processed by this pipeline """
            return self._classes

        # Public Interface

        def loadSamples(self,
                        sampleIdentifiers: np.ndarray,
                        refLabelFeaturePair: LabelFeaturePair,
                        toLoadIndex: int) -> None:
            """ Load the sample corresponding to the identifiers """
            refLabelFeaturePair.features[toLoadIndex] = \
                np.empty(shape=(sampleIdentifiers.size,self._numFeatures),dtype=np.float32)
            for ii,(x,y) in enumerate(zip(sampleIdentifiers,refLabelFeaturePair.labels)):
                refLabelFeaturePair.features[toLoadIndex][ii] = self.__loadSample(x,y)
            return None

        def loadClass(self,
                      classIdentifier: int,
                      refLabelFeaturePair: LabelFeaturePair,
                      toLoadIndex: int) -> None:
            """ Load all samples corresponding to the provided class """

            sampleIdentifiers = self.__findAllSamplesFromClass()
            refLabelFeaturePair.features[toLoadIndex] = \
                np.empty(shape=(sampleIdentifiers.size,self._numFeatures),dtype=np.float32)
            for ii,(x,y) in enumerate(zip(sampleIdentifiers,refLabelFeaturePair.labels)):
                refLabelFeaturePair.features[toLoadIndex][ii] = self.__loadSample(x,y)
            return None

        # Private Interface

        def __logMessage(self, message: str) -> None:
            """ Log message via the parent dataset """
            self._parentDataset.logMessage(message)
            return None

        def __loadSample(self,sampleID: int,targetID: int) -> np.ndarray:
            """ Load a single sample from a specified class """
            toLoad = self._getPath(targetID,sampleID)
            X = np.fromfile(toLoad,dtype=np.float32)
            return X

        def __findAllSamplesFromClass(self) -> np.ndarray:
            """ Return all of the samples in the class """



            return None

        def __loadShapes(self) -> None:
            """ Load in all of the shapes """
            shapesFile = os.path.join(self._rootPath,"featureShapes.txt")
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
            self._shuffled      = np.arange(self._database.size,dtype=int)
            self._shuffledIter  = 0

        def __del__(self):
            """ Destructor """
            pass

        # Accessors

        def getSize(self) -> int:
            """ Return the actual number of samples """
            return self._actualSize

        def getTargets(self, sampleIDs: np.ndarray) -> np.ndarray:
            """ Return the targets corresponding to the provided samples """
            return self._database[sampleIDs]

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

        def shuffle(self, randomSeed: int) -> None:
            """ Shuffles the order that samples will be drawn. Resets the internal iterator """
            self._shuffled = np.arange(self._database.size)
            if (randomSeed != 0):
                np.random.seed(randomSeed)
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

        def getNext(self,count=1) -> np.ndarray:
            """ Draw the next 'count' number of samples """            
            upperBound = np.min([self._shuffledIter + count,self._database.size])
            actualCount = upperBound - self._shuffledIter
            result = np.ones(shape=(actualCount,),dtype=int) * -1
            result = np.copy(self._shuffled[self._shuffledIter:upperBound]) # samples to draw
            self._shuffledIter = (self._shuffledIter + actualCount) % self._database.size
            return result

        # Private Interface

        def __logMessage(self, message: str) -> None:
            """ Log message via the parent dataset """
            self._parentDataset.logMessage(message)
            return None

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
        pipelineCounter = 0
        for item in rootContents:
            fullRoot = os.path.join(self._rootPath,item)
            if (os.path.isdir(fullRoot) == False):
                continue
            if (item.startswith(PIPELINE) == False):
                continue
            self.__logMessage("Adding {0} to list of pipelines".format(item))
            pipeline = Dataset.__Pipeline(fullRoot,self,pipelineCounter)
            self._pipelines.append(pipeline)
            pipelineCounter += 1
        return None

