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

        #### CLASS DEFINITIONS ####

class AppSettings:
    """ Stores runtime settings and configuration """
    
    def __init__(self,
                 inputPaths: list,
                 outputPath: str):
        """ Constructor """
        self._pathsInput    = set()     # __initInputPaths
        self._pathOutput    = "NULL"    # __initOutputPath
        self._loggerName    = "textLog.txt"

        self._collectionThreads     = 1
        self._findFilesRecurseDepth = 1

        self._maxSampleFetchAttempts = 16
        self._waitFetchDuration = 8000

        self._developmentMode = True

        self.__initInputPaths(inputPaths)
        self.__initOutputPath(outputPath)

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getInputPaths(self) -> list:
        """ Return a list of sample input paths """
        return list(self._pathsInput)

    def getOutputPath(self) -> str:
        """ Return output Path """
        return self._pathOutput

    def getTextLogPath(self) -> str:
        """ Retun the path where the text logger is outputted to """
        return os.path.join(self._pathOutput,self._loggerName)

    def getNumCollectionThreads(self) -> int:
        """ Return the number of collection threads to use """
        return self.getNumCollectionThreads

    def getFindFilesRecursionDepth(self) -> int:
        """ Retur the depth use to recurse directory trees """
        return self._findFilesRecurseDepth
     
    def getMaxSampleFetchAttempts(self) -> int:
        """ Returns the max number of times a thread should try to pull a sample """
        return self._maxSampleFetchAttempts

    def getWaitFetchDuration(self) -> float:
        """ Return the time in ms between fetch attempts """
        return self._waitFetchDuration

    def getIsDevMode(self) -> bool:
        """ Return T/F if this run is in development mode """
        return self._developmentMode

    # Public Interface

    @staticmethod
    def developmentSettings():
        """ Return settings used for development """
        return None

    # Private Interface

    def __initInputPaths(self,
                        listOfInputPaths: list) -> None:
        """ Initialize provided input paths """
        for path in listOfInputPaths:
            absPath = os.path.abspath(path)
            if (os.path.isdir(absPath) == True):
                self._pathsInput.add(absPath)
            elif (os.path.isfile(absPath) == True):
                self._pathsInput.add(absPath)
            else:
                # No folder or file
                pass
        listOfInputPaths.clear()
        return None
     
    def __initOutputPath(self,
                         outputPath: str) -> None:
        """ Initialize provided output path """
        outputPath = os.path.abspath(outputPath)
        msg = ""
        if (os.path.isdir(outputPath) == True):
            # Is a directory
            msg = "Provided output path: {0} already exists. Contents may be overwritten".format(outputPath)
        elif (os.path.isfile(outputPath) == True):
            # Is a file
            msg = "Provided output path: {0} already exists. Contents may be overwritten".format(outputPath)
        else:
            # Is a directory
            msg = "Generating new output path at: {0}".format(outputPath)
        print(msg)
        self._pathOutput = outputPath
        return None