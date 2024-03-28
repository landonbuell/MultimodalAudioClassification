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

        #### CLASS DEFINITIONS ####

class ComponentManager:
    """ Abstract Base Class for application components """

    def __init__(self,
                 componentName: str,
                 collectionApp):
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

    def getApp(self):
        """ Return the feature collection app """
        return self._app

    def getSettings(self):
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


class ManagerDatabase:
    """ Stores a collection of managers from parent application """

    def __init__(self,
                 app):
        """ Constructor """
        self._app = app

    def __del__(self):
        """ Destructor """
        self._app = None

    # Accessors

    def getAppSettings(self):
        """ Return a ref to the app settings """
        return self._app.getSettings()

    def getSampleDatabase(self):
        """ Return a ref to the sample database """
        return self._app.getSampleDatabase()

    def getPipelineManager(self):
        """ Return a ref to the pipeline manager """
        return self._app.getPipelineManager()

    def getCollectionManager(self):
        """ Return a ref to the collection manager """
        return self._app.getCollectionManager()

    # Magic Methods

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0} @ {1} -> {2}".format(
            self.__class__,hex(id(self)),repr(self._app))