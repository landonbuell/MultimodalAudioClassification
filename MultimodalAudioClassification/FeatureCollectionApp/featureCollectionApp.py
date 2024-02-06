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

        #### CLASS DEFINITIONS ####

class FeatureCollectionApplication:
    """ Encapsulates the functionality of the Feature Collection Application """

    __instance = None

    def __init__(self):
        """ Constructor """
        self.__registerSelfAsSingleton()

        self._status = 0
        

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

    # Public Interface

    def run(self) -> int:
        """ Run the application """
        self.__startup()
        if (self._status == 0):
            self.__execute()
        self.__execute()
        return self._status

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
        return None

    def __execute(self) -> None:
        """ Run the execution """
        return None

    def __cleanup(self) -> None:
        """ Run the cleanup """
        return None

    # Magic Methods

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))
