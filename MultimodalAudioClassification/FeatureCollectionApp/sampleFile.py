"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       sampleFile.py
    Classes:    SampleFileIO

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import os
import scipy.io.wavfile as sciowav

import signalData


        #### CLASS DEFINITIONS ####

class SampleFileIO:
    """ Stores basic info for each sample file """

    def __init__(self,
                 targetClass: int,
                 sourceFile: str,
                 channelIndex=0):
        """ Constructor """
        self._target    = targetClass
        self._source    = os.path.abspath(sourceFile)

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getTarget(self):
        """ Return class target """
        return self._target

    def getSource(self) -> str:
        """ Return the file source """
        return self._source

    def isReal(self) -> bool:
        """ Return T/F if source file exists """
        return os.path.isfile(self._source)

    # Public Interface

    def decode(self) -> list:
        """ decode the source file into a list of signalData instances """
        listOfSignalDatas = []
        if (self._source.endswith(".txt") == True):
            # Is Text File
            listOfSignalDatas = self.__decodeTxtFile()
        elif (self._source.endswith(".wav") == True):
            # Is a WAV file
            listOfSignalDatas = self.__decodeWavFile()
        else:
            # Cannot handle this type
            listOfSignalDatas = [signalData.SignalData(targetClass=self._target),]
        return listOfSignalDatas

    # Private Interface

    def __decodeTxtFile(self) -> list:
        """ Decode a .txt file into a signal data instance """
        return [signalData.SignalData(targetClass=self._target),]

    def __decodeWavFile(self) -> list:
        """ Decode a .wav file into a signal data instance """
        (sampleRate,channels) = sciowav.read(self._source)
        numChannels = channels.shape[1]
        signals = []
        for ii in range(numChannels):
            newSignal = signalData.SignalData(
                sampleRate=sampleRate,
                targetClass=self._target,
                waveform=channels[ii,:].transpose,
                sourcePath=self._source,
                channelIndex=ii)
            signals.append(newSignal)
        return self
