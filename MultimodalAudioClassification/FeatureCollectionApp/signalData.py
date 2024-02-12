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

    # Constructors 

    def __init__(self,
                 sampleRate=44100,
                 targetClass=-1,
                 sourcePath="NULL_PATH"):
        """ Constructor """
        self._sampleRate    = sampleRate
        self._targetClass   = targetClass
        self._waveform      = np.zeros(size=(16,),dtype=np.float32)
        self._cachedData    = SignalData.CachedData()
        self._sourcePath    = sourcePath
        self._channelIndex  = 0

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

    @property
    def waveform(self):
        """ Return the waveform as a property """
        return self._waveform

    def getCachedData(self) -> CachedData:
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

    # Public Interface

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

