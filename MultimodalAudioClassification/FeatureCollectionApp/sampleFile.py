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
import numpy as np
import scipy.io.wavfile as sciowav

import signalData


        #### CLASS DEFINITIONS ####

class SampleFileIO:
    """ Stores basic info for each sample file """

    __uniqueIdentifier = 0

    def __init__(self,
                 targetClass: int,
                 sourceFile: str):
        """ Constructor """
        self._target    = targetClass
        self._source    = os.path.abspath(sourceFile)
        self._id        = SampleFileIO.__uniqueIdentifier
        SampleFileIO.__uniqueIdentifier += 1

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

    def getId(self) -> int:
        """ Return the sample ID """
        return self._id

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
        (sampleRate,wavData) = sciowav.read(self._source)   
        signals = []
        # wavData.shape = (Nsamples, Nchannels)

        if (wavData.ndim == 1):
            # 1D Array - single channel
            newSignal = signalData.SignalData(
                sampleRate=sampleRate,
                targetClass=self._target,
                waveform=wavData,
                sourcePath=self._source,
                channelIndex=0)
            signals.append(newSignal)

        elif (wavData.ndim == 2):
            # 2D Array - Multiple channels
            numChannels = wavData.shape[1]
            signals = [None] * numChannels
            for ii in range(numChannels):
                channelData = np.squeeze(wavData[:,ii])
                newSignal = signalData.SignalData(
                    sampleRate=sampleRate,
                    targetClass=self._target,
                    waveform=channelData,
                    sourcePath=self._source,
                    channelIndex=ii)
                signals.append(newSignal)

        else:
            msg = "Cannot handle signal w/ {0} axes".format(wavData.ndim)
            raise RuntimeError(msg)

        return signals


    # Magic Methods

    def __str__(self) -> str:
        """ Cast to string """
        return "Sample #{0} @ {1}".format(self._id,self._source)

    def __repr__(self) -> str:
        """ Representation for debugging """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))

    