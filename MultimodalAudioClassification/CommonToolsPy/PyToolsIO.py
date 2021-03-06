"""
Repository:     MultimodalAudioClassification
Solution:       MultimodalAudioClassification
Project:        CommonToolsPy
File:           PyToolsIO.py
 
Author:         Landon Buell
Date:           June 2022
"""

        #### IMPORTS ####

import os
import sys

import numpy as np

        #### CLASS DEFINITIONS ####

class Serializer:
    """ Abstract Base Class for all Serializer Classes """

    def __init__(self,data,path):
        """ Constructor for Serializer Abstract Class """
        self._data              = data
        self._outputPath        = path

        if (os.path.exists(path) == True):
            msg = "\tWARNING - Overwriting object at {0}".format(path)
            print(msg)

        self._outFileStream     = None
        self._outFmtStr         = lambda key,val :  "{0:<32}\t{1:<128}\n".format(key,val)

    def __del__(self):
        """ Destructor for Serializer Abstract Class """
        self._data = None
        if (self._outFileStream is not None):
            self._outFileStream.close()
        return

    # Public Interface

    def call(self):
        """ Write Object to OutputStream """
        return False

    # Protected Interface

    @staticmethod
    def listToString(inputList,delimiter=","):
        """ Convert Elements of list to string w/ delimiter """
        outputString = ""
        if len(inputList) == 0:
            # No Items in the Input List
            outputString += "-1,"
        else:
            # Items in the Input List
            for item in inputList:
                outputString += str(item) + delimiter
        return outputString.strip()

    def findAndParseStrs(self,keyword):
        """ Find All words with token and return as list of Strings"""
        result = []
        for line in self._inFileContents:
            tokens = line.split()
            if tokens[0].startswith(keyword):
                result.append(tokens[-1].strip())
        return result

    def findAndParseInts(self,keyword):
        """ Find All words with token and return as list of Strings"""
        result = self.findAndParseStrs(keyword)
        result = result[0].split(',')
        result = ["".join(ch for ch in x if ch.isdigit()) for x in result]
        result = [int(x) for x in result if x != '']
        return result

    def writeHeader(self):
        """ Add Header To Output """
        self._outFileStream.write(self.__repr__() + "\n")
        self._outFileStream.write("-"*64 + "\n")
        return self

    def writeFooter(self):
        """ Add Header To Output """
        self._outFileStream.write("-"*64 + "\n")
        self._outFileStream.write(self.__repr__() + "\n")
        return self

    # Magic Methods

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

class Deserializer:
    """ Abstract Base Class for all Deserializer Classes """

    def __init__(self,path):
        """ Constructor for Deserializer Abstract Class """
        self._data              = None
        self._inputPath         = path

        if (os.path.exists(path) == False):
            msg = "\tWARNING - No readable object at {0}".format(path)
            raise FileNotFoundError(msg)

        self._inFileStream      = None

    def __del__(self):
        """ Destructor for Deserializer Abstract Class """
        self._data = None
        if (self._inFileStream is not None):
            self._inFileStream.close()
        return

    # Public Interface

    def call(self):
        """ Read Object From inputStream """

        return False

    @staticmethod
    def stringToList(inputString,delimiter=" ",outType=None):
        """ Convert string to list of type """
        outputList = inputString.split(delimiter)
        if outType is not None:
            outputList = [outType(x) for x in outputList]
        return outputList

    # Magic Methods

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

class AppSettingsSerializer(Serializer):
    """ Class to Serialize AppSettings Instance """

    def __init__(self,data,path):
        """ Constructor AppSettingsSerializer """
        super().__init__(data,path)

    def __del__(self):
        """ Destructor for AppSettingsSerializer """
        super().__del__()

    def call(self):
        """ Serialize the Chosen Instance """

        self._outFileStream = open(self._outputPath,"w")
        self.writeHeader()
        # Write In/Out P    aths
            
        self._outFileStream.write( self._outFmtStr("startupPath",self._data.getStartupPath() ) )
        for i,path in enumerate(self._data.getInputPaths()):
            self._outFileStream.write( self._outFmtStr("InputPath_" + str(i),path ) )
        self._outFileStream.write( self._outFmtStr("outputPath",self._data.getOutputPath() ) )

        # Write Collection Settings
        self._outFileStream.write( self._outFmtStr("BatchSize",self._data.getBatchSize() ) )
        self._outFileStream.write( self._outFmtStr("BatchLimit",self._data.getBatchLimit() ) )
        self._outFileStream.write( self._outFmtStr("ShuffleSeed",self._data.getShuffleSeed() ) )
        self._outFileStream.write( self._outFmtStr("Verbose",self._data.getVerbose() ) )

        # Write Log Levels
        self._outFileStream.write( self._outFmtStr("LogConsole",self._data.getLogToConsole() ) )
        self._outFileStream.write( self._outFmtStr("LogFile",self._data.getLogToFile() ) )

        # Close Output + Return
        self.writeFooter()
        self._outFileStream.close()
        return self

class DesignMatrixDataSerializer(Serializer):
    """ Class to Serialize the 'X' portion of a Design Matrix """

    def __init__(self,data,path):
        """ Constructor for DesignMatrixDataSerializer """
        super().__init__(data,path)

    def __del__(self):
        """ Destructor for DesignMatrixDataSerializer """
        super().__del__()

    # Public Interface

    def call(self):
        """ Write Object to Output Stream """

        # Get the Number of Samples + Raw Feature Data
        X = self._data.getFeatures()
        X = X.flatten().tobytes()

        # Open the Output Stream (write bytes)
        self._outFileStream = open(self._outputPath,"wb")
        self._outFileStream.write( X );

        # Close + Return
        self._outFileStream.close()
        return True

class DesignMatrixLabelSerializer(Serializer):
    """ Class to Serialize the 'Y' portion of a Design Matrix """

    def __init__(self,data,path):
        """ Constructor for DesignMatrixLabelSerializer """
        super().__init__(data,path)

    def __del__(self):
        """ Destructor for DesignMatrixLabelSerializer """
        super().__del__()

    # Public Interface

    def call(self):
        """ Write Object to Output Stream """

        # Get the Number of Samples + Raw Feature Data
        Y = self._data.getFeatures()
        Y = Y.flatten().tobytes()

        # Open the Output Stream (write bytes)
        self._outFileStream = open(self._outputPath,"wb")
        self._outFileStream.write( Y );

        # Close + Return
        self._outFileStream.close()
        return True

class RunInfoSerializer(Serializer):
    """ Clas to Serialize RunInfo Structure """

    def __init__(self,data,path):
        """ Constructor for RunInfoSerializer """
        super().__init__(data,path)

    def __del__(self):
        """ Destructor for RunInfoSerializer """
        super().__del__()

    # Public Interface

    def call(self):
        """ Write Object to OutputStream """
        self._outFileStream = open(self._outputPath,"w")
        self.writeHeader()

        # Write Input/OutputPaths
        self._outFileStream.write( self._outFmtStr("OutputPath",self._data.getOutputPath() ))
        self.writeListAsRows( "InputPath", self._data.getInputPaths() )

        # Write Pipeline Data

        # Close and Return
        self.writeFooter()
        self._outFileStream.close()
        return


    # Private Interface

    def writeListAsRows(self,listName,listContents):
        """ Write Each Element of the List to the output Buffer """
        for idx,item in enumerate(listContents):
            lhs = listName + "[" + idx + "]"
            rhs = str(listContents)
            row = self._outFmtStr(lhs,rhs)
            self._outFileStream.write( row )
        return self


    


