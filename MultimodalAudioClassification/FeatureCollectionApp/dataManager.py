"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       dataManager.py
    Classes:    RundataManager

    Author:     Landon Buell
    Date:       April 2024
"""

        #### IMPORTS ####

import componentManager

        #### CLASS DEFINITIONS ####

class RundataManager(componentManager.ComponentManager):
    """ Stores high-level data related to the current run """

    __NAME = "RundataManager"

    def __init__(self,
                 app):
        """ Constructor """
        super().__init__(RundataManager.__NAME)
        self._runInfo   = None
        self._classInfo = None

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getRunInfo(self) -> object:
        """ Return a ref to the run info structure """
        return self._runInfo

    def getClassInfo(self) -> object:
        """ Return a ref to the class info structure """
        return self._classInfo

    # Public Interface

    def initialize(self) -> None:
        """ OVERRIDE: Initialize the Sample Database """
        super().initialize()
        
        return None

    def teardown(self) -> None:
        """ OVERRIDE: Teardown the Sample Database """
        super().teardown()
        self.__exportRunInfo()
        self.__exportClassInfo()
        return None

    def registerSample(self, sampleData) -> None:
        """ Register this sample w/ the data manager """

        return None

    # Private Interface

    def __initRunInfo(self) -> None:
        """ Initialize the run info structure """
        return None

    def __initClassInfo(self) -> None:
        """ Initialize the class Info Struct """
        return None

    def __exportRunInfo(self) -> None:
        """ Export the run info struct to the output path """
        return None

    def __exportClassInfo(self) -> None:
        """ Export the class info struct to the output path """
        return None
