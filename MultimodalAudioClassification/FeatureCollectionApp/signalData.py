"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       signalData.py
    Classes:    SignalData, 
                SignalData.CachedData

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import os
import numpy as np

        #### CLASS DEFINITIONS ####

class SignalData:
    """ Stores information related to a single Waveform """

    class CachedData:
        """ Stores Additional Signal Data information rather than recompute it """

        def __init__(self):
            """ Constructor """
            self.analysisFramesTime = None
            self.analysisFramesFreq = None

        def __del__(self):
            """ Destructor """
            pass

    # Applies a unique label to each sample
    __unqiueCounter = 0

    # Constructors 

    def __init__(self,
                 sampleRate=44100,
                 targetClass=-1,
                 waveform=None,
                 sourcePath="NULL_PATH",
                 channelIndex=0):
        """ Constructor """
        self._sampleRate    = sampleRate
        self._targetClass   = targetClass
        self._waveform      = waveform
        self._cachedData    = SignalData.CachedData()
        self._sourcePath    = sourcePath
        self._channelIndex  = channelIndex
        self._uniqueID      = SignalData.__unqiueCounter

        SignalData.__unqiueCounter += 1
        if (self._waveform is None):
            self._waveform = np.array(shape=(1,),dtype=np.float32)

    def __del__(self):
        """ Destructor """
        self._cachedData.clear()

    # Accessors

    def getSampleRate(self) -> int:
        """ Return the sample rate for this signal """
        return self._sampleRate

    def setSampleRate(self,
                      sampleRate: int) -> None:
        """ Set the sample rate for this signal """
        self._sampleRate = sampleRate
        return None

    def getTarget(self) -> int:
        """ Return the target class for this signal """
        return self._targetClass

    def setTarget(self,
                  targetClass: int) -> None:
        """ Set the tatget class for this signal """
        self._targetClass = targetClass
        return None

    def getWaveform(self) -> np.ndarray:
        """ Return the waveform """
        return self._waveform

    def setWaveform(self,
                    waveform: np.ndarray) -> None:
        """ Set the waveform """
        self._waveform = waveform
        return None

    @property
    def waveform(self):
        """ Return the waveform as a property """
        return self._waveform

    @property
    def cachedData(self) -> CachedData:
        """ Return the underlying Cached data for this signal """
        return self._cachedData

    def getSourcePath(self) -> str:
        """ Return the path that this signal originated from """
        return self._sourcePath

    def setSourcePath(self,
                      sourcePath: str) -> None:
        """ Set the path that this signal originated from """
        self._sourcePath = sourcePath
        return None

    def getChannelIndex(self) -> int:
        """ Return the channel index that this signal originted from """
        return self._channelIndex

    def setChannelIndex(self,
                        channelIndex: int) -> None:
        """ Set the channel index that this signal originated from """
        self._channelIndex = channelIndex
        return None

    def getNumSamples(self) -> int:
        """ Return the number of samples in this waveform """
        return self._waveform.size

    @property
    def uniqueID(self) -> int:
        """ Return the unique counter ID for this signal """
        return self._uniqueID

    # Public Interface

    def exportPathBinary(self,
                         extension="bin"):
        """ location within the output path where to export this sample """
        classFolder = "class{0}".format(self._targetClass)
        fileName = "sample{0}.{1}".format(self.uniqueID,extension)
        return os.path.join(pipelineFolder,classFolder,fileName)

    def clearCachedData(self) -> None:
        """ Clear the underlying cached data """
        # TODO: Implement this!
        return None

    def populateTimeSeriesAnalysisFrames(self) -> None:
        """ Populate the cached data' time series analysis frames """
        # TODO: Implement this!
        return None

    def populateFreqSeriesAnalysisFrames(self) -> None:
        """ Populate the cached data' frequency series analysis frames """
        # TODO: Implement this!
        return None

    # Public Interface

    # Private Interface

    # Magic Methods

    def __getitem__(self,
                    key: int) -> np.ndarray:
        """ Return the frame at the provided value """
        return self._waveform[key]

    def __setitem__(self,
                    key: int,
                    val: np.ndarray) -> None:
        """ Set the provided value at the provided frame key """
        self._waveform[key] = val
        return None

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))

