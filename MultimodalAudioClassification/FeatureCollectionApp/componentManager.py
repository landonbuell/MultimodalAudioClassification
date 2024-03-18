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

import featureCollectionApp

        #### CLASS DEFINITIONS ####

class ComponentManager:
    """ Abstract Base Class for application components """

    def __init__(self,
                 componentName: str,
                 collectionApp: featureCollectionApp.FeatureCollectionApplication):
        """ Constructor """
        self._name  = componentName
        self._app   = collectionApp

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getName(self) -> str:
        """ Return name of this component """
        return self._name

    def getApp(self) -> featureCollectionApp.FeatureCollectionApplication:
        """ Return the feature collection app """
        return self._app

    def getSettings(self) -> featureCollectionApp.appSettings.AppSettings:
        """ Return the app's settings """
        return self._app.getSettings()

    # Public Interface

    def logMessage(self,
                   message: str) -> None:
        """ Log message through parent app """
        self._app.logMessage(message)
        return None
   
    def initialize(self) -> None:
        """ Initialize this component for use """
        msg = "Initializing {0} ...".format(self._name)
        self.logMessage(msg)
        return None

    def teardown(self) -> None:
        """ Teardown this component from use """
        msg = "Tearing down {0} ...".format(self._name)
        self.logMessage(msg)
        return None

    # Protected Interface

    # Magic Methods

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))