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

import enum
import queue

import componentManager

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

    class DequeueResult:
        """ Is the result of getting the front of the sample database """

        def __init__(self,
                     sample,
                     status):
            """ Constructor """
            self.sampleData     = sample
            self.databaseStatus = status

        def __del__(self):
            """ Destructor """
            pass

    def __init__(self,
                 app: componentManager.featureCollectionApp.FeatureCollectionApplication):
        """ Constructor """
        super().__init__(SampleDatabase.__NAME,app)
        self._inputFiles    = queue.Queue()
        self._database      = queue.Queue(SampleDatabase.__CAPACITY)      
        self._size          = 0
        self._password      = ""

    def __del__(self):
        """ Destructor """
        self._inputFiles    = None
        self._database      = None

    # Accessors

    def getSize(self) -> int:
        """ Return the current size of the database """
        self._size

    def getCapacity(self) -> int:
        """ Return the capacity of the database """
        self._database.maxsize()

    def isFull(self) -> bool:
        """ Return T/F if the database is full """
        return self._database.full()

    def isEmpty(self) -> bool:
        """ Return T/F if the database if empty """
        return self._database.empty()

    def isLocked(self) -> bool:
        """ Return T/F if the database is locked """
        return self._password != ""

    # Public Interface

    def initialize(self) -> None:
        """ OVERRIDE: Initialize the Sample Database """
        super().initialize()
        return None

    def teardown(self) -> None:
        """ OVERRIDE: Teardown the Sample Database """
        super().teardown()
        return None

    def lock(self,password: str) -> bool:
        """ Attempt to lock the database. Return T/F is successful """
        if (self.isLocked() == True):
            return False
        # Is not locked yet
        self._password = password
        return True

    def unlock(self, password: str) -> bool:
        """ Attempt to unlock the database. Return T/F if successful """
        if (self.isLocked() == False):
            return True
        if (self._password == password):
            # passwords match
            self._password = ""
            return True
        # Passwords do not match
        return False

    def getNext(self,sample) -> int:
        """ Enqueue a sample to the database """
        if (self.isLocked() == True):
            return SampleDatabase.Status.DATABASE_LOCKED
        if (self.isEmpty() == True):
            return SampleDatabase.Status.DATABASE_EMPTY
        sample = self._database.get()
        self._size += 1
        return SampleDatabase.Status.DATABASE_STABLE

    # Private Interface

    def __buildInputFileQueue(self) -> None:
        """ Build a queue of input files to read for sample files """
        return None

    def __readInputFiles(self) -> None:
        """ Read all input files to collect sample files """
        return None
    
    def __enqueueSample(self,sample) -> int:
        """ Enqueue a sample to the database """
        if (self.isLocked() == True):
            return SampleDatabase.Status.DATABASE_LOCKED
        if (self.isFull() == True):
            return SampleDatabase.Status.DATABASE_FULL
        self._database.put(sample)
        self._size += 1
        return SampleDatabase.Status.DATABASE_STABLE






