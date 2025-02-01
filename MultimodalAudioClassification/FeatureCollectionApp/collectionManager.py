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

import datetime

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
        self._timeStart     = datetime.datetime.min
        self._timeFinish    = datetime.datetime.max
        self.__initCollectors()

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def numCollectors(self) -> int:
        """ Return the number of feature collectors """
        return len(self._collectors)

    def getCollectionStartTime(self) -> datetime.datetime:
        """ Return the time when collection began """
        return self._timeStart

    # Public Interface

    def initialize(self) -> None:
        """ OVERRIDE: Initialize the collection manager """
        super().initialize()     
        return None

    def teardown(self) -> None:
        """ OVERRIDE: Teardown the collection manager """
        super().teardown()
        self.__logGrossTimeElapsed()
        return None

    def runCollection(self) -> int:
        """ Run the Collection Session """    
        self._timeStart = datetime.datetime.now()
        self.__startCollection()  
        self._timeFinish = datetime.datetime.now()
        return self._status

    def logMessage(self,
                    message: str) -> None:
        """ Log Message to the parent app """
        self._app.logMessage(message)
        return None

    # Private Interface

    @staticmethod
    def getCurrentTimeStamp():
        """ Get the current time in YYYY.MM.DD.HH.MM.SS.UUUUUU """
        now = str(datetime.datetime.now())
        now = now.replace(" ",".")
        now = now.replace(":",".")
        now = now.replace("-",".")
        return str(now)

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

    def __logGrossTimeElapsed(self) -> None:
        """ Log time current elapsed since collection began """
        if (self._timeFinish == datetime.datetime.max):
            self._timeFinish = datetime.datetime.now()
        timeDelta = self._timeFinish - self._timeStart
        msg = "Collection time elapsed: {0}".format(timeDelta)
        return None