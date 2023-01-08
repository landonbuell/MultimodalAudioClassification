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
        self._outFileStream     = None
        self._outFmtStr         = lambda key,val :  "{0:<32}\t{1:<128}\n".format(key,val)

    def __del__(self):
        """ Destructor for Serializer Abstract Class """
        if (self._outFileStream is not None):
            self._outFileStream.close()
        return

    # Public Interface

    def call(self):
        """ Write Object to OutputStream """

        return False
    
    # Protected Interface

    def _writeHeader(self):
        """ Add Header To Output """
        self._outFileStream.write(repr(self) + "\n")
        self._outFileStream.write("-"*64 + "\n")
        return self

    def _writeFooter(self):
        """ Add Header To Output """
        self._outFileStream.write("-"*64 + "\n")
        self._outFileStream.write(repr(self) + "\n")
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

    # Protected Interface

    # Static Interface

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