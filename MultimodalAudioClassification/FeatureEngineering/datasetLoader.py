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

class DatasetLoader:
    """ Loads samples from a  """

    DEFAULT_SIZE = int(2**10)

    def __init__(self,
                parentDataset: object):
        """ Constructor """
        self._parentDataset = parentDataset
        self._database      = np.ones(shape=(256,),dtype=int) * -1
        self._orderedIter   = 0
        self._shuffled      = np.arange(self._database.size,dtype=int)
        self._shuffledIter  = 0

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getRoot(self) -> str:
        """ Return the root path of the dataset """
        return self._parentDataset.getRoot()

    def getTargets(self, sampleIDs: np.ndarray) -> np.ndarray:
        """ Return the targets corresponding to the provided samples """
        return self._database[sampleIDs]

    # Public Interface

    def populate(self,shuffleSeed=0) -> None:
        """ Populate this database """
        sampleFiles = self.__findSampleFiles()
        for item in sampleFiles:
            self.__countSamplesInFile(os.path.join(self.getRoot(),item))
        if (shuffleSeed != 0):
            self.shuffle(shuffleSeed)
        return None

    def shuffle(self, randomSeed: int) -> None:
        """ Shuffles the order that samples will be drawn. Resets the internal iterator """
        self._shuffled = np.arange(self._database.size)
        if (randomSeed != 0):
            np.random.seed(randomSeed)
            np.random.shuffle(self._shuffled)
        self._shuffledIter = 0
        return None

    def resetOrderedIter(self) -> None:
        """ Reset the internal ordered iterator """
        self._orderedIter = 0
        return None

    def resetRandomIter(self) -> None:
        """ Reset the internal random iterator """
        self._randomIter = 0
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

    def getIndexOfNextOrdered(self,sampleCount: int) -> np.ndarray:
        """ Return the index of the next 'sampleCount' samples in order"""
        upperBound = np.min([self._orderedIter + sampleCount,self._database.size])
        indicesOfNext = np.arange(self._orderedIter,upperBound,dtype=int) # samples to draw      
        self._orderedIter = (self._orderedIter + indicesOfNext.size) % self._database.size
        return indicesOfNext

    def getIndexOfNextRandom(self,sampleCount: int) -> np.ndarray:
        """ Return the index of the next 'sampleCount' samples in the shuffled order"""          
        upperBound = np.min([self._shuffledIter + sampleCount,self._database.size])
        indicesOfNext = self._shuffled[self._shuffledIter:upperBound] # samples to draw      
        self._shuffledIter = (self._shuffledIter + indicesOfNext.size) % self._database.size
        return indicesOfNext

    def getIndexOfAllFromClass(self,classIndex: int) -> np.ndarray:
        """ Return the index of all samples in a chosen class """
        whereClassMatches = np.where(self._database == classIndex)[0]
        return whereClassMatches

    def load(self,  
             sampleIDs: np.ndarray, 
             targetIDs: np.ndarray,
             pipelineID: int) -> np.ndarray:
        """ Load a sample for this pipeline """
        ptrPipeline = self._parentDataset.getPipeline(pipelineID)
        matrixShape = (sampleIDs.size,ptrPipeline.getNumFeatures())
        X = np.empty(shape=matrixShape,dtype=np.float32)
        # Load In each sample
        for ii,(sample,target) in enumerate(zip(sampleIDs,targetIDs)):
            path = ptrPipeline.getSamplePath(sample,target)
            x = np.fromfile(path,dtype=np.float32)
            if (x.size != ptrPipeline.getNumFeatures()):
                msg = "Size mismatch in reading features for sample #{0} in pipeline {1}".format(
                    sample,self._pipelineName)
                raise RuntimeError(msg)
            X[ii] = x
        return X

    # Private Interface

    def __findSampleFiles(self) -> list:
        """ Load all of the sample files in this data set """
        rootContents = os.listdir(self.getRoot())
        DOT_TXT = ".txt"
        SAMPLES = "samples"
        sampleFiles = list()
        for item in rootContents:
            if (os.path.isfile(os.path.join(self.getRoot(),item)) == False):
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
                self._database[sampleIndex] = classIndex
        return None

    # MAGIC METHODS

    def __getitem__(self,key: int):
        """ Index an item in the database """
        return self._database[key]

    def __setitem__(self,key: int, val: int):
        """ Set an item in the database """
        self._database[key] = val
        return None