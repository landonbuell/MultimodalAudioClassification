"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       componentManager.py
    Classes:    ComponentManager

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import os
import enum
import queue

import componentManager
import sampleFile

import sampleGenerator

        #### CLASS DEFINITIONS ####

class SampleDatabase(componentManager.ComponentManager):
    """ Stores a collection of samples to process """

    __NAME = "SampleDatabase"
    __CAPACITY = int(2**16)

    class Status(enum.IntEnum):
        """ Gives the status of the database """
        STABLE = 0
        LOCKED = 1
        FULL   = 2
        EMPTY  = 3

    def __init__(self,
                 app):
        """ Constructor """
        super().__init__(SampleDatabase.__NAME,app)
        self._inputFiles    = queue.Queue()
        self._generators    = queue.Queue()
        self._database      = queue.Queue(app.getSettings().getSampleLimit())      
        self._size          = 0     # also tracks size
        self._queued        = 0     # total number of samples queued
        self._released      = 0     # total number of samples released

    def __del__(self):
        """ Destructor """
        self._inputFiles    = None
        self._generators    = None
        self._database      = None

    # Accessors

    def getSize(self) -> int:
        """ Return the current size of the database """
        self._size

    def getNumQueued(self) -> int:
        """ Return the total number of samples queued """
        return self._queued

    def getNumReleased(self) -> int:
        """ Return the total number of samples released """
        self._released

    def getCapacity(self) -> int:
        """ Return the capacity of the database """
        self._database.maxsize()

    def isFull(self) -> bool:
        """ Return T/F if the database is full """
        return self._database.full()

    def isEmpty(self) -> bool:
        """ Return T/F if the database if empty """
        return self._database.empty()

    # Public Interface

    def initialize(self) -> None:
        """ OVERRIDE: Initialize the Sample Database """
        super().initialize()
        self.__buildSampleDatabaseFromInputFiles()
        self.__buildSampleDatabaseFromGenerators()
        self.logState()
        if (len(self) == 0):
            # No Samples Added
            msg = "No saamples added to database"
            self.logMessage(msg)       
        return None

    def teardown(self) -> None:
        """ OVERRIDE: Teardown the Sample Database """   
        super().teardown()
        self.logState()
        return None

    def getNext(self) -> sampleFile.SampleFileIO:
        """ Return the next sample in the queue """
        result = self._database.get()
        self._size -= 1
        self._released += 1
        return result

    def logState(self) -> None:
        """ Log the current state of the sample manager """
        msg = "{0} is populated with ({1}/{2}) samples".format(
            self,self._size,self._database.maxsize)
        self.logMessage(msg)
        msg = "{0} has released ({1}/{2}) queued samples".format(
            self, self._released,self._queued)
        self.logMessage(msg)
        return None

    # Private Interface

    def __buildSampleDatabaseFromInputFiles(self) -> None:
        """ Build a queue of input files to read for sample files """
        listOfInputFiles = self.getSettings().getInputPaths()
        if (len(listOfInputFiles) == 0):
            return None
        for item in listOfInputFiles:
            if (self.isFull() == True):
                break
            fullPathToItem = os.path.abspath(item)
            if (os.path.isdir(fullPathToItem) == True):
                # Item is a Directory 
                self.__handleInputDirectory(fullPathToItem,0)
            elif (os.path.isfile(fullPathToItem) == True):
                # Item is a file
                self.__handleInputFile(fullPathToItem)
            else:
                # Item is an unknown type
                pass
        return None

    def __buildSampleDatabaseFromGenerators(self) -> None:
        """ Build a queue of input files to read for sample files """
        listOfDataGenerators = self.getSettings().getDataGenerators()
        if (len(listOfDataGenerators) == 0):
            return None
        for generator in listOfDataGenerators:
            self.__handleSampleGenerator(generator)
        return None

    def __handleInputDirectory(self,dirpath: str, currentDepth: int) -> None:
        """ Handle reading a directory """
        maxDepth = self.getSettings().getFindFilesRecursionDepth()
        if (currentDepth >= maxDepth):
            return None
        contents = os.listdir(dirpath)
        for item in contents:
            fullPathToItem = os.path.abspath(item)
            if (os.path.isdir(fullPathToItem) == True):
                # Item is a Directory 
                self.__handleInputDirectory(fullPathToItem,currentDepth + 1)
            elif (os.path.isfile(fullPathToItem) == True):
                # Item is a file
                self.__handleInputFile(fullPathToItem)
            else:
                # Item is an unknown type
                pass
            # If full, exit
            if (self.isFull() == True):
                break
        return None

    def __handleInputFile(self, filePath: str) -> None:
        """ Handle reading a single input file """
        line = ""
        lineCounter = 0
        success = True
        with open(filePath,"r") as inputStream:
            while (success == True):
                line = inputStream.readline()
                lineCounter += 1
                if (lineCounter == 1):
                    continue
                if (not line):
                    break                
                line = line.strip()
                # Try to make a sample file instance 
                try:
                    lineTokens = line.split(",")
                    newSample = sampleFile.SampleFileIO(int(lineTokens[1]),lineTokens[0])
                    targetClassName = lineTokens[2]
                except Exception as err:
                    msg = "Could note sample SampleFileIO from {0}, line#{1} for reason: {2}".format(
                        filePath,lineCounter,str(err))
                    self.logMessage(msg)
                    continue
                # If the instance worked, add it to the queue
                success = self.__enqueueSample(newSample,targetClassName)
        return None
    
    def __handleSampleGenerator(self,
                                generator: sampleGenerator.SampleGenerator) -> None:
        """ Read and store all samples from the provided generator """
        classIndex  = generator.getClassIndex()
        className   = generator.getClassName()

        while(generator.isEmpty() == False):

            if (self.isFull() == True):
                msg = "Sample Database is full. {0} samples remain in {1}".format(
                    generator.drawsRemaining(), str(generator))
                self.logMessage(msg)
                break

            generatedSample = generator.drawNext()
            generatedSampleFile = sampleFile.GeneratedSampleFileIO(
                classIndex,
                generatedSample.sampleRate,
                generatedSample.waveform)
            self.__enqueueSample(generatedSampleFile,className)

        return None

    def __enqueueSample(self,
                        sample: sampleFile.SampleFileIO,
                        targetName: str) -> int:
        """ Enqueue a sample to the database """
        if (self.isFull() == True):
            return False
        dataManager = self.getApp().getDataManager()
        dataManager.registerExpectedSample(
            sample.getTarget(), targetName)
        self._database.put(sample)
        self._size += 1
        self._queued += 1
        return True

    # DUNDER METHODS

    def __len__(self) -> int:
        """ Return the number of samples """
        return self._size
    






