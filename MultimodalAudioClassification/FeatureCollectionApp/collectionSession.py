"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionApp
    File:       collectionSession.py
    Classes:    CollectionSession

    Author:     Landon Buell
    Date:       March 2024
"""


        #### IMPORTS ####

import os
import threading

import featureCollectionApp
import featureCollector

        #### CLASS DEFINITIONS ####

class FeatureCollectionSession:
    """ Encapsulates a collection session """

    def __init__(self,
                 app,
                 numThreads: int):
        """ Constructor """
        self._app           = app
        self._collectors    = [None] * numThreads
        self._status        = 0

        self.__initCollectors()

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getSampleDatabase(self):
        """ Return a ref to the sample database """
        return self._app.getSampleDatabase()

    def getPipelineManager(self):
        """ Return a ref to the pipeline manager """
        return self._app.getPipelineManager()

    def getOutputPath(self) -> str:
        """ Return the top level output path """
        return self._app.getSettings().getOutputPath()

    @property
    def numCollectors(self) -> int:
        """ Return the number of feature collectors """
        return len(self._collectors)

    # Public Interface

    def run(self) -> int:
        """ Run the Collection Session """
        featureCollector.FeatureCollector.registerSession(self)
        self.__startCollection()  
        featureCollector.FeatureCollector.deregisterSession()
        return self._status

    def logMessage(self,
                    message: str) -> None:
        """ Log Message to the parent app """
        self._app.logMessage(message)
        return None

    # Private Interface

    def __initCollectors(self) -> None:
        """ Initialize all feature collectors """
        for ii in range(self._collectors):
            name = "collector{0}".format(ii)
            self._collectors[ii] = featureCollector.FeatureCollector(name)
        return None

    def __startCollection(self) -> None:
        """ Begin the collection process """
        if (self.numCollectors == 1):
            # Single thread
            self._collectors[0].run()
        else:
            # Multi-thread
            for collector in self._collectors:
                collector.start()
            for collector in self._collectors:
                collector.join()
        return None