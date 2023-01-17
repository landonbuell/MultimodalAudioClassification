"""
Repository:     SignalClassifierApp
Solution:       SignalClassifierApp
Project:        CommonUtilitiesPy
File:           PyToolsIO.py
 
Author:         Landon Buell
Date:           December 2022
"""

        #### IMPORTS ####

import os
import sys

import numpy as np

        #### CLASS DEFINITIONS ####

class Serializer:
    """ Abstract Base Class for all Serializer Classes """

    fmtKeyValPair = lambda x,y : "{0:<32}{1}\n".format(x,y)

    def __init__(self,data,path):
        """ Constructor for Serializer Abstract Class """
        self._data              = data
        self._outputPath        = path
        self._buffer            = []

    def __del__(self):
        """ Destructor for Serializer Abstract Class """
        self._data = None
        self._buffer.clear()

    # Public Interface

    def appendLine(self,line):
        """ Append single line to output buffer """
        self._buffer.append(line)
        return self

    def appendLines(self,lines):
        """ Append multiple lines to output buffer """
        for item in lines:
            self._buffer.append(item)
        return self

    def call(self):
        """ ABSTRACT: Write Object to OutputStream """
        return True
    
    # Protected Interface

    def _writeBufferToPath(self):
        """ Write Output Buffer to File """
        outFileStream = open(self._outputPath,"w")
        for line in self._buffer:
            self._buffer.write(line + "\n")
        outFileStream.close()
        return self

    def _writeHeader(self):
        """ Add Header To Output """
        self.appendLine( repr(self) + "\n" )
        self.appendLine( "-"*64 + "\n" )
        return self

    def _writeFooter(self):
        """ Add Footer To Output """
        self.appendLine( "-"*64 + "\n" )
        self.appendLine( repr(self) + "\n" )
        return self

    # Static Interface

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

    # Magic Methods

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

class Deserializer:
    """ Abstract Base Class for all Deserializer Classes """

    def __init__(self,path):
        """ Constructor for Deserializer Abstract Class """
        self._data      = None
        self._inputPath = path
        self._buffer    = []

    def __del__(self):
        """ Destructor for Deserializer Abstract Class """
        self._data = None
        self._buffer.clear()
        return

    # Public Interface

    def call(self):
        """ Read Object From inputStream """
        inFileStream = open(self._inputPath,'r')
        self._buffer = inFileStream.readlines()
        inFileStream.close()
        return True

    # Protected Interface

    def _findInBuffer(self,key):
        """ Find Key in Buffer """
        vals = []
        for item in self._buffer:
            if (item.startswith(key)):
                tokens = item.split()
                vals.append( tokens[-1] )
        return vals

    # Static Interface

    @staticmethod
    def stringToList(inputString,delimiter=" ",outType=None):
        """ Convert string to list of type """
        outputList = inputString.split(delimiter)
        if (outputList[-1] == ""):
            outputList.pop()
        if outType is not None:
            outputList = [outType(x) for x in outputList]
        return outputList

    @staticmethod
    def stringToInt(inputString):
        """ Convert string to integer """
        result = 0
        try:
            result = int(inputString.strip())
        except Exception as err:
            print(err)
        return result

    @staticmethod
    def stringToBool(inputString):
        """ Determine if item is T/F """
        TRUE    = ["TRUE","T","1","YES","Y"]
        FALSE   = ["FALSE","F","0","NO","N"]
        text = inputString.upper()
        if (text in TRUE):
            return True
        else:
            return False

    # Magic Methods

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))