"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       zeroCrossingRate.py
    Classes:    TotalZeroCrossingRate,
                ZeroCrossesPerFrame,

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import numpy as np

import collectionMethod

        #### CLASS DEFINITIONS ####

class TotalZeroCrossingRate(collectionMethod.AbstractCollectionMethod):
    """
        Divide a waveform into N segments and compute the energy in each
    """

    __NAME = "TotalZeroCrossingRate"
    __NUM_FEATURES = 1

    def __init__(self):
        """ Constructor """
        super().__init__(TotalZeroCrossingRate.__NAME,
                         TotalZeroCrossingRate.__NUM_FEATURES)

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Protected Interface

    def _callBody(self,
                  signal: signalData.SignalData):
        """ OVERRIDE: Compute TDE's for signal """
        waveformSign = np.sign(signal.waveform)
        waveformDiff = np.diff(waveformSign)
        self._data[0] = np.sum(waveformDiff) / 2.0
        return True

class FrameZeroCrossingRate(collectionMethod.AbstractCollectionMethod):
    """
        Divide a waveform into N segments and compute the energy in each
    """

    __NAME = "TotalZeroCrossingInfo"
    __NUM_FEATURES = 6

    def __init__(self):
        """ Constructor """
        super().__init__(FrameZeroCrossingRate.__NAME,
                         FrameZeroCrossingRate.__NUM_FEATURES)
        self._callbacks.append( collectionMethod.CollectionMethodCallbacks.signalHasAnalysisFramesTime )

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Public Interface

    def featureNames(self) -> list:
        """ VIRTUAL: Return a list of the feature names """
        result = ["FrameZeroCrossingRateMean",
                  "FrameZeroCrossingRateVariance",
                  "FrameZeroCrossingRateMedian",
                  "FrameZeroCrossingRateMin",
                  "FrameZeroCrossingRateMax",
                  "FrameZeroCrossingRateRange"]
        return result
        
    # Protected Interface

    def _callBody(self,
                  signal: signalData.SignalData):
        """ OVERRIDE: Compute TDE's for signal """
        zxrs = np.empty(shape=(signal.cachedData.analysisFramesTime.maxNumFrames,),dtype=np.float32)
        for ii in range(signal.cachedData.analysisFramesTime.maxNumFrames):
            zxrs[ii] = self.__computeZeroCrossingRateOfFrame(signal,ii)
        # Store values
        self._data[0]   = np.mean(zxrs)
        self._data[1]   = np.var(zxrs)
        self._data[2]   = np.median(zxrs)
        self._data[3]   = np.min(zxrs)
        self._data[4]   = np.max(zxrs)
        self._data[5]   = self._data[4] - self._data[3]
        return True

    # Private Interface

    def __computeZeroCrossingRateOfFrame(self,
                                         signal: signalData.SignalData,
                                         frameIndex: int) -> None:
        """ Compute the zero crossing rate for a chosen frame """
        sign = np.sign(signal.cachedData.analysisFramesTime[frameIndex])
        diff = np.diff(sign)
        return np.sum(diff) / 2.0
