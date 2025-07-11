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

class AppSettings:
    """ Stores runtime settings and configuration """
    
    def __init__(self,
                 inputPaths: list,
                 generators: list,
                 outputPath: str,
                 findFileRecurseDepth=1,
                 numCollectionThreads=1,
                 sampleLimit=int(2**16),
                 samplesPerFile=1024):
        """ Constructor """
        self._pathsInput        = set()     # __initInputPaths
        self._generators        = set()     # __initDataGenerators
        self._pathOutput        = "NULL"    # __initOutputPath
        self._loggerName        = "textLog"
        self._developmentMode   = True

        self._findFilesRecurseDepth = findFileRecurseDepth
        self._collectionThreads     = numCollectionThreads  
        self._sampleLimit           = sampleLimit
        self._samplesPerFile        = samplesPerFile    # samples per output file

        self.__initInputPaths(inputPaths)
        self.__initOutputPath(outputPath)

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getInputPaths(self) -> list:
        """ Return a list of sample input paths """
        return list(self._pathsInput)

    def getDataGenerators(self) -> list:
        """ Return a list of data generator instances """
        return list(self._generators)

    def getOutputPath(self) -> str:
        """ Return output Path """
        return self._pathOutput

    def getTextLogPath(self,includeTimeStamp=True) -> str:
        """ Retun the path where the text logger is outputted to """
        loggerPath = os.path.join(self._pathOutput,self._loggerName + ".txt")
        if (includeTimeStamp == True):
            now = AppSettings.getCurrentTimeStamp()
            loggerPath = os.path.join(self._pathOutput,self._loggerName + now + ".txt")
        return loggerPath

    def getIsDevMode(self) -> bool:
        """ Return T/F if this run is in development mode """
        return self._developmentMode

    def getFindFilesRecursionDepth(self) -> int:
        """ Retur the depth use to recurse directory trees """
        return self._findFilesRecurseDepth

    def getNumCollectionThreads(self) -> int:
        """ Return the number of collection threads to use """
        return max(self._collectionThreads,1)
 
    def getSampleLimit(self) -> int:
        """ Return the limit on the number of samples to read """
        return min(self._sampleLimit,int(2**24))

    def getSamplesPerOutputFile(self) -> int:
        """ Return the number of samples per sample#.txt file """
        return self._samplesPerFile
     
    @staticmethod
    def getCurrentTimeStamp():
        """ Get the current time in YYYYMMDDHHMMSS"""
        now = str(datetime.datetime.now())
        now = now.replace(" ","")
        now = now.replace(":","")
        now = now.replace("-","")
        now = now.replace(".","")
        now = now[:-6]
        return str(now)

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

    # Magic Methods

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))