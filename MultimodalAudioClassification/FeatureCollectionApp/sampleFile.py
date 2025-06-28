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

    def isValid(self) -> bool:
        """ VIRTUAL: Return T/F if source file exists """
        return os.path.isfile(self._source)

    def getExtension(self) -> str:
        """ Return the extension at the end of the file path """
        return self._source.split(".")[-1]

    # Public Interface

    def getSignals(self) -> list:
        """ VIRTUAL: Decode the source file into a list of signalData instances """
        listOfSignalDatas = []
        extension = self.getExtension()
        if (extension == "txt"):
            # Is Text File
            listOfSignalDatas = self.__decodeTxtFile()
        elif (extension == "wav"):
            # Is a WAV file
            listOfSignalDatas = self.__decodeWavFile()
        elif (extension == "bin"):
            listOfSignalDatas = self._
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

    def __decodeBinFile(self) -> list:
        """ Decode a .bin file into a signal data instance """


    # Magic Methods

    def __str__(self) -> str:
        """ Cast to string """
        return "Sample #{0} @ {1}".format(self._id,self._source)

    def __repr__(self) -> str:
        """ Representation for debugging """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))
    
class GeneratedSampleFileIO(SampleFileIO):
    """ Represents a sample file that is NOT on disk, but stashed in a variable """

    __GENERATED_SAMPLE = "GENERATED_SAMPLE"

    def __init__(self,
                 targetClass: int,
                 sampleRate: int,
                 sourceData: np.ndarray,
                 ):
        """ Constructor """
        super().__init__(targetClass,
                         GeneratedSampleFileIO.__GENERATED_SAMPLE)
        self._rate  = sampleRate
        self._data  = sourceData

    def __del__(self):
        """ Destructor """
        super().__del__()

    class GeneratedSampleParams:
        """ Stores parameters used to generate the attatched waveform """


    # ACCESSORS 

    def isValid(self) -> bool:
        """ OVERRIDE: Return T/F if the source data is usable """
        return ((self._data.shape[-1] > 1024) and (self._data.ndim <= 2))
    
    # PUBLIC 

    def getSignals(self) -> list:
        """ VIRTUAL: Decode the source file into a list of signalData instances """
        listOfSignalDatas = []
        if (self._data.ndim == 1):
            newSignal = signalData.SignalData(sampleRate=self._rate,
                                      targetClass=self.getTarget(),
                                      waveform=self._data,
                                      sourcePath=GeneratedSampleFileIO.__GENERATED_SAMPLE,
                                      channelIndex=0)
            listOfSignalDatas.append(newSignal)
        elif (self._data.ndim == 2):
            # Each row is a new waveform
            for ii in range(self._data.shape[0]):
                newSignal = signalData.SignalData(sampleRate=self._rate,
                                          targetClass=self.getTarget(),
                                          waveform=self._data[ii],
                                          sourcePath=GeneratedSampleFileIO.__GENERATED_SAMPLE,
                                          channelIndex=ii)
                listOfSignalDatas.append(newSignal)
        else:
            msg = "Cannot handle generated sample data with more than two axes"
            raise RuntimeError(msg)
        return listOfSignalDatas