"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       featureCollectionApp.py
    Classes:    FeatureCollectionApp

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import os
import sys

import appSettings

import sampleDatabase
import pipelineManager
import collectionManager

import textLogger # CommonToolsPy

        #### CLASS DEFINITIONS ####

class FeatureCollectionApplication:
    """ Encapsulates the functionality of the Feature Collection Application """

    __instance = None

    def __init__(self,
                 settings: appSettings.AppSettings):
        """ Constructor """
        self.__registerSelfAsSingleton()
        if (os.path.isdir(settings.getOutputPath()) == False):
            os.makedirs(settings.getOutputPath())

        self._status    = 0
        self._settings  = settings
        self._logger    = textLogger.TextLogger(settings.getTextLogPath())

        self._sampleDatabase    = sampleDatabase.SampleDatabase(self)
        self._pipelineManager   = pipelineManager.PipelineManager(self)
        self._collectionManager = collectionManager.CollectionManager(self)

    def __del__(self):
        """ Destructor """
        FeatureCollectionApplication.__instance = None

    @staticmethod
    def getInstance():
        """ Return the singleton instance """
        if (FeatureCollectionApplication.__instance is None):
            msg = "An instance of FeatureCollectionApplication does not exist!"
            raise RuntimeError(msg)
        return FeatureCollectionApplication.__instance

    # Accessors

    def getStatus(self) -> int:
        """ Return the internal status """
        return self._status

    def getSettings(self) -> appSettings.AppSettings:
        """ Return a ref to the app settings """
        return self._settings

    def getSampleDatabase(self) -> sampleDatabase.SampleDatabase:
        """ Return a ref to the sample database """
        return self._sampleDatabase

    def getPipelineManager(self) -> pipelineManager.PipelineManager:
        """ Return a ref to the pipeline manager """
        return self._pipelineManager

    def getCollectionManager(self) -> collectionManager.CollectionManager:
        """ Return a ref to the collection manager """
        return self._collectionManager

    # Public Interface

    def logMessage(self,message: str) -> None:
        """ Log a message to the text logger """
        self._logger.logMessage(message)
        return None

    def run(self) -> int:
        """ Run the application """
        self.__startup()
        if (self._status == 0):
            self.__execute()
        self.__cleanup()
        return self._status

    def logMessage(self,
                   message : str) -> None:
        """ Log a message """
        self._logger.logMessage(message)
        return None

    # Private Interface

    def __registerSelfAsSingleton(self) -> None:
        """ Register this instance as the singleton, throw error if already exists """
        if (FeatureCollectionApplication.__instance is not None):
            msg = "An instance of FeatureCollectionApplication already exists!"
            raise RuntimeError(msg)
        FeatureCollectionApplication.__instance = self
        return None

    def __startup(self) -> None:
        """ Run the startup """
        self._sampleDatabase.initialize()
        self._pipelineManager.initialize()
        self._collectionManager.initialize()
        return None

    def __execute(self) -> None:
        """ Run the execution """
        self._collectionManager.runCollection()
        return None

    def __cleanup(self) -> None:
        """ Run the cleanup """
        self._sampleDatabase.teardown()
        self._pipelineManager.teardown()
        self._collectionManager.teardown()
        return None

    # Magic Methods

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))
