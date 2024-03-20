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

        #### CLASS DEFINITIONS ####

class FeatureCollectionSession:
    """ Encapsulates a collection session """

    def __init__(self,
                 app: featureCollectionApp.FeatureCollectionApplication,
                 numThreads: int):
        """ Constructor """
        self._app           = app
        self._collectors    = [None] * numThreads
        self._status        = 0

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getApp(self) -> featureCollectionApp.FeatureCollectionApplication:
        """ Return the owning app """
        return self._app

    # Public Interface

    def run(self) -> int:
        """ Run the Collection Session """

        return self._status

    # Protected Interface

    def _logMessage(self,
                    message: str) -> None:
        """ Log Message to the parent app """
        self._app.logMessage(message)
        return None