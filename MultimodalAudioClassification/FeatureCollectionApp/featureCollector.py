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
import signalData

        #### CLASS DEFINITIONS ####

class FeatureCollector(threading.Thread):
    """ Represents an object that processes a sample """

    # Stores a static fef to the active collection session
    __activeCollectionSession = None

    def __init__(self,
                 name: str):
        """ Constructor """
        super().__init__(group=None,target=None,name=name)
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

    # Static interface

    @staticmethod
    def registerSession(session: collectionSession.FeatureCollectionSession) -> None:
        """ Register the active collection session """
        FeatureCollector.__activeCollectionSession = session
        return None

    @staticmethod
    def deregisterSession() -> None:
        """ Register the active collection session """
        FeatureCollector.__activeCollectionSession = None
        return None

    @staticmethod
    def getSession() -> collectionSession.FeatureCollectionSession:
        """ Return a ref to the collection session """
        if (FeatureCollector.__activeCollectionSession is None):
            msg = "Attempting to acces non-existant collection session"
            raise RuntimeError(msg)
        return FeatureCollector.__activeCollectionSession

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
        sample = self.__pullNextSampleUntilValid()
        if (sample is None):
            return False
        listOfSignals = sample.decode()
        for signal in listOfSignals:
            listOfFeatureVectors = self.__collectFeatures(signal)
            self.__exportFeatures(signal,listOfFeatureVectors)
        return True

    # Private Interface

    def __getNextSample(self):
        """ Get the next sample from the sample database """
        if (FeatureCollector.getSession().getSampleDatabase().isEmpty() == False):
            return FeatureCollector.getSession().getSampleDatabase().getNext()
        return None

    def __pullNextSampleUntilValid(self):
        """ Get the next sample from the sample database """
        nextSample = None
        while (True):
            nextSample = self.__getNextSample()
            if (nextSample is None):
                break
            if (nextSample.isReal() == True):
                break
        return nextSample

    def __collectFeatures(self,
                         signal: signalData.SignalData) -> list:
        """ Send Signal through each pipeline """
        pipelineMgr = FeatureCollector.getSession().getApp().getPipelineManager()
        listOfFeatureVectors = pipelineMgr.processSignal(signal)
        return listOfFeatureVectors

    def __exportFeatures(self,
                         signal: signalData.SignalData,
                         listOfFeatureVectors: list) -> None:
        """ Export Feature vectors to appropriate output locations """
        parentOutputPath = FeatureCollector.getSession().getOutputPath()
        signalOutputPath = signal.exportPathBinary()
        for ii,vector in enumerate(listOfFeatureVectors):
            if (vector is None):
                continue
            pipelineOutputPath = "pipeline{0}".format(ii)
            fullOutputPath = os.path.join(parentOutputPath,pipelineOutputPath,signalOutputPath)
            vector.toBinaryFile(fullOutputPath)
        return None