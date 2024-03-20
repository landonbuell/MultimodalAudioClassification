"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionApp
    File:       featureCollector.py
    Classes:    FeatureCollector

    Author:     Landon Buell
    Date:       March 2024
"""


        #### IMPORTS ####

import os
import threading

import collectionSession

        #### CLASS DEFINITIONS ####

class FeatureCollector(threading.Thread):
    """ Represents an object that processes a sample """

    def __init__(self,
                 session: collectionSession.FeatureCollectionSession,
                 name: str):
        """ Constructor """
        super().__init__(group=None,target=None,name=name)
        self._session   = session
        self._stopEvent = threading.Event()

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getName(self) -> str:
        """ Return the name of this thread """
        return self.name

    def getIdentifier(self) -> int:
        """ Return the thread id """
        return self.native_id

    def isAlive(self) -> bool:
        """ Returns T/F if this thread is alive """
        return self.is_alive()

    def isStopped(self) -> bool:
        """ Return T/F is the stop flag has been raised """
        return self._stopEvent.is_set()

    # Public Interface

    def run(self) -> None:
        """ OVERRIDE: Represents the threads execution logic """        
        while (self.isStopped() == False):
            success = self.processNext()
            if (success == False):
                self._stopEvent.set()
        return None

    def terminate(self) -> None:
        """ Raise an internal 'stop' this thread event """
        self._stopEvent.set()
        return None

    def processNext(self) -> bool:
        """ Collect features for a single sample """
        sample = self.__pullNextSample()

        return 

    # Private Interface

    def __pullNextSample(self):
        """ Get the next sample from the sample database """
        sample = self._session.getApp().getSampleDatabase().getNext()
        return sample
