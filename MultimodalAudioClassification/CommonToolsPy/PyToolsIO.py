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
        self._outFileBuffer     = []

    def __del__(self):
        """ Destructor for Serializer Abstract Class """
        self._data = None
        self._outFileBuffer.clear()

    # Public Interface

    def appendLine(self,line):
        """ Append single line to output buffer """
        self._outFileBuffer.append(line)
        return self

    def appendLines(self,lines):
        """ Append multiple lines to output buffer """
        for item in lines:
            self._outFileBuffer.append(item)
        return self

    def call(self):
        """ ABSTRACT: Write Object to OutputStream """
        return True
    
    # Protected Interface

    def _writeBufferToPath(self):
        """ Write Output Buffer to File """
        outFileStream = open(self._outputPath,"w")
        for line in self._outFileBuffer:
            self._outFileBuffer.write(line + "\n")
        outFileStream.close()
        return self

    def _writeHeader(self):
        """ Add Header To Output """
        lines = [repr(self),"-"*64]
        self.appendLines( lines )
        return self

    def _writeFooter(self):
        """ Add Header To Output """
        lines = ["-"*64,repr(self)]
        self.appendLines( lines )
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
        self._data              = None
        self._inputPath         = path
        self._inFileBuffer      = []

    def __del__(self):
        """ Destructor for Deserializer Abstract Class """
        self._data = None
        if (self._inFileStream is not None):
            self._inFileStream.close()
        return

    # Public Interface

    def call(self):
        """ Read Object From inputStream """
        inFileStream = open(self._inputPath,'r')
        self._inFileBuffer = inFileStream.readlines()
        inFileStream.close()
        return True

    # Protected Interface

    def _findInBuffer(self,key):
        """ Find Key in Buffer """
        vals = []
        for item in self._inFileBuffer:
            if (item.startswith(key)):
                tokens = item.split()
                vals.append( tokens[-1] )
        return vals

    # Static Interface

    @staticmethod
    def stringToList(inputString,delimiter=" ",outType=None):
        """ Convert string to list of type """
        outputList = inputString.split(delimiter)
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
        text = inputString.capitalize()
        if (text in TRUE):
            return True
        else:
            return False

    # Magic Methods

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))