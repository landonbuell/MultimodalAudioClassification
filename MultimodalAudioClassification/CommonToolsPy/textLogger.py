"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    CommonToolsPy
    File:       textLogger.py
    Classes:    TextLogger

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import os
import datetime

        #### CLASS DEFINITIONS ####

class TextLogger:
    """ Represents a text logger """

    def __init__(self,
                 outputPath: str,
                 toConsole=True):
        """ Constructor """
        self._outputPath = os.path.abspath(outputPath)
        self._logCounter = 0
        self._logToConsole = toConsole
       
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getOutputPath(self) -> str:
        """ Return logger output path """
        return self._outputPath

    def getLogToConsole(self) -> bool:
        """ Return T/F if we should log to console """
        return self._logToConsole

    # Public Interface
    
    def logMessage(self,
                     message: str) -> None:
        """ Log a message """
        timeStamp = TextLogger.__getCurrentTimeStamp()
        formattedMessage = "\t{0:<8}{1:<32}{2}".format(self._logCounter,timeStamp,message)
        if (self._logToConsole):
            print(formattedMessage)
        self.__writeToFile(formattedMessage)
        return None

    # Private Interface

    def __writeToFile(self,message:str) -> None:
        """ Write a message to the output file """
        with open(self._outputPath,"a") as outputStream:
            outputStream.write(message + "\n")
        self._logCounter += 1
        return None

    @staticmethod
    def __getCurrentTimeStamp():
        """ Get the current time in YYYY.MM.DD.HH.MM.SS.UUUUUU """
        now = str(datetime.datetime.now())
        now = now.replace(" ",".")
        now = now.replace(":",".")
        now = now.replace("-",".")
        return str(now)

    # Magic Methods

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))



