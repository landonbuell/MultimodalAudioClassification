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
import datetime
import numpy as np
import threading

import featureCollectionApp
import sampleFile

        #### CLASS DEFINITIONS ####

class FeatureCollector(threading.Thread):
    """ Represents an object that processes a sample """


    def __init__(self,
                 name: str):
        """ Constructor """
        super().__init__(group=None,target=None,name=name)
        self._callbackGetNext   = GetNextSampleStrategies.getNextSampleSingleThread
        self._stopEvent         = threading.Event()
        self._databaseKey       = str(id(self)) # temp use the instance's memory address

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

    def stopFlag(self) -> bool:
        """ Return T/F is the stop flag has been raised """
        return self._stopEvent.is_set()


    # Public Interface

    def run(self) -> None:
        """ OVERRIDE: Represents the threads execution logic """      
        # Run collection
        while (self.stopFlag() == False):
            self.__processNext()
        return None

    def logMessage(self,message: str) -> None:
        """ Log a Message to the collection manager """
        message = self.getName() + ": " + message
        featureCollectionApp.FeatureCollectionApplication.getInstance().logMessage(message)
        return None

    def raiseStopFlag(self,reason=None) -> None:
        """ Raise an internal 'stop' this thread event """
        if (reason is None):
            reason = "No reason provided"
        message = "stop flag raised for {0} for reason: {1}".format(
            self.getName(),reason)
        self.logMessage(message)
        self._stopEvent.set()
        return None

    # Private Interface

    def __processNext(self) -> bool:
        """ Collect features for a single sample """
        if (self.stopFlag() == False):
            # Pull Next Sample
            nextSample = self.__invokeGetNextSample()
            if (nextSample is None):
                return False
            self.__logNextSample(nextSample)
        if (self.stopFlag() == False):
            # Decode Sample into list of signals
            listOfSignals = self.__decodeSample(nextSample)
            if (len(listOfSignals) == 0):
                return False
        if (self.stopFlag() == False):
            # Process List of Signals + Export Features
            self.__processListOfSignals(listOfSignals)
        return True

    def __invokeGetNextSample(self) -> sampleFile.SampleFileIO:
        """ Pull Next Sample from sample database """
        if (self._callbackGetNext is None):
            self.raiseStopFlag(reason="No strategy for getting next sample is selected")
            return None
        while(True):
           nextSample = self._callbackGetNext.__call__(self)
           if (nextSample is None):
               return None
           if (nextSample.isValid() == True):
               return nextSample
        return None

    def __logNextSample(self, nextSample: sampleFile.SampleFileIO) -> None:
        """ Log the next sample """
        collectionManager = featureCollectionApp.FeatureCollectionApplication.getInstance().getCollectionManager()
        timeDelta = datetime.datetime.now() - collectionManager.getCollectionStartTime()
        msg = "Pulled sample: {0}. Time: {1}".format(
            str(nextSample),str(timeDelta))
        collectionManager.logMessage(msg)
        return None

    def __decodeSample(self, sampleFile) -> list:
        """ Decode the sample and return a list of signals """
        try:
            listOfSignals = sampleFile.getSignals()
            msg = "Read {0} into {1} signals".format(sampleFile,len(listOfSignals))
            self.logMessage(msg)
            return listOfSignals
        except Exception as err:
            msg = "Failed to read signals from {0} due to error: {1}".format(
                str(sampleFile),str(err))
            self.logMessage(msg)
        return []

    def __processListOfSignals(self, listOfSignals: list) -> None:
        """ Process list of signals and export feature vectors """
        pipelineMgr = featureCollectionApp.FeatureCollectionApplication.getInstance().getPipelineManager()
        for signal in listOfSignals:
            # Process signals and get list of Features for each pipeline
            signal = self.__preprocessSignal(signal)
            msg = "Sending {0} to be processed by pipeline manager".format(signal)
            self.logMessage(msg)
            listOfFeatureVectors = pipelineMgr.processSignal(signal)
            self.__exportListOfFeatureVectors(signal,listOfFeatureVectors)
        return None

    def __preprocessSignal(self, signal) -> object:
        """ Preprocess a single signal """
        # Cast to new type
        signal.normalizeAmplitude(np.float32)
        #signal.showWaveform()
        return signal

    def __exportListOfFeatureVectors(self, 
                                     signal: object,
                                     listOfFeatureVectors: list) -> None:
        """ Export a list of feature Vectors to binaries """
        dataManager = featureCollectionApp.FeatureCollectionApplication.getInstance().getDataManager()
        failureCount = 0
        for ii,vector in enumerate(listOfFeatureVectors):
            # Export
            if ((vector is None) or (len(vector) == 0)):
                msg = "Got None for feature vector on signal {0}, pipeline {1}".format(
                    signal.uniqueID(),ii)
                self.logMessage(msg)
                continue
            # Get output Path
            outputLocation = dataManager.getExportLocation(ii,signal.getTarget())
            fullOutputPath = os.path.join(outputLocation,signal.exportNameBinary())
            # Export
            try:
                vector.toBinaryFile(fullOutputPath)
                msg = "Exported sample #{0} to {1}".format(signal.uniqueID(),fullOutputPath)
            except RuntimeError as err:
                msg = str(err)
                failureCount += 1
            except Exception as err:
                 msg = "Failed to export sample #{0} to {1}".format(signal.uniqueID(),fullOutputPath)
                 failureCount += 1
            self.logMessage(msg)
        # All done!
        dataManager.registerExportedSample(signal.getTarget())
        return None

class GetNextSampleStrategies:
    """ Static class of callbacks for getting the next from sample database """

    @staticmethod
    def getNextSampleSingleThread(collector: FeatureCollector) -> object:
        """ Pull the next sample from the sample database while on only main thread """
        sampleDatabase = featureCollectionApp.FeatureCollectionApplication.getInstance().getSampleDatabase()
        if (sampleDatabase.isEmpty() == True):
            # Database is empty
            collector.raiseStopFlag(reason="Sample database is empty")
            return None
        return sampleDatabase.getNext()

    @staticmethod
    def getNextSampleMultiThread(collector: FeatureCollector) -> object:
        """ Pull the next sample from the sample database in a multithreaded environment """
        sampleDatabase = featureCollectionApp.FeatureCollectionApplication.getInstance().getSampleDatabase()
        if (sampleDatabase.isEmpty() == True):
            # Database is empty
            collector.raiseStopFlag(reason="Sample database is empty")
            return None
        # TODO: Fix this for multiple threads
        return sampleDatabase.getNext()