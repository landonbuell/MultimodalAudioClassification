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

import componentManager
import featureCollector

        #### CLASS DEFINITIONS ####

class CollectionManager(componentManager.ComponentManager):
    """ Stores and execute feature collectors """

    __NAME = "CollectionManager"

    def __init__(self,
                 app):
        """ Constructor """
        super().__init__(CollectionManager.__NAME,app)
        self._collectors = [None] * app.getSettings().getNumCollectionThreads()
        self.__initCollectors()

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def numCollectors(self) -> int:
        """ Return the number of feature collectors """
        return len(self._collectors)

    # Public Interface

    def initialize(self) -> None:
        """ OVERRIDE: Initialize the collection manager """
        super().initialize()     
        featureCollector.FeatureCollector.registerManagerDatabase(self.getApp())
        return None

    def teardown(self) -> None:
        """ OVERRIDE: Teardown the collection manager """
        super().teardown()
        return None

    def runCollection(self) -> int:
        """ Run the Collection Session """       
        self.__startCollection()  
        return self._status

    def logMessage(self,
                    message: str) -> None:
        """ Log Message to the parent app """
        self._app.logMessage(message)
        return None

    # Private Interface

    def __initCollectors(self) -> None:
        """ Initialize all feature collectors """
        for ii in range(self.numCollectors):
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