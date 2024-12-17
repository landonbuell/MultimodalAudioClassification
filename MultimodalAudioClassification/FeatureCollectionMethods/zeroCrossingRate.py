"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       zeroCrossingRate.py
    Classes:    TotalZeroCrossingRate,
                ZeroCrossesPerFrame,

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import numpy as np

import collectionMethod
import analysisFrames
import callbacks

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
                  signal: collectionMethod.signalData.SignalData,
                  features: collectionMethod.featureVector.FeatureVector):
        """ OVERRIDE: Compute TDE's for signal """
        waveformSign = np.sign(signal.waveform)
        waveformDiff = np.abs(np.diff(waveformSign)) * 0.5
        zeroCrossingRate = np.sum(waveformDiff) / signal.numSamples
        features.appendItem(zeroCrossingRate)
        return True

class FrameZeroCrossingRate(collectionMethod.AbstractCollectionMethod):
    """
        Divide a waveform into N segments and compute the energy in each
    """

    __NAME = "TotalZeroCrossingInfo"
    __NUM_FEATURES = 6

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters):
        """ Constructor """
        super().__init__(FrameZeroCrossingRate.__NAME,
                         FrameZeroCrossingRate.__NUM_FEATURES)
        self._params = frameParams

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
                  signal: collectionMethod.signalData.SignalData,
                  features: collectionMethod.featureVector.FeatureVector):
        """ OVERRIDE: Compute TDE's for signal """
        signal.makeFreqSeriesAnalysisFrames(self._params)
        numFramesInUse = signal.cachedData.analysisFramesTime.getNumFramesInUse()
        zxrs = np.zeros(shape=(numFramesInUse,),dtype=np.float32)
        for ii in range(numFramesInUse):
            zxrs[ii] = self.__computeZeroCrossingRateOfFrame(signal,ii)
        # 
        # Store values
        zxrInfo = np.empty(shape=(self.getNumFeatures(),),dtype=np.float32)
        zxrInfo[0] = np.mean(zxrs)
        zxrInfo[1] = np.var(zxrs)
        zxrInfo[2] = np.median(zxrs)
        zxrInfo[3] = np.min(zxrs)
        zxrInfo[4] = np.max(zxrs)
        zxrInfo[5] = np.max(zxrs) - np.min(zxrs)

        # Add to feature vector
        features.appendItems(zxrInfo)
        return True

    # Private Interface

    def __computeZeroCrossingRateOfFrame(self,
                                         signal: collectionMethod.signalData.SignalData,
                                         frameIndex: int) -> None:
        """ Compute the zero crossing rate for a chosen frame """
        frame = signal.cachedData.analysisFramesTime[frameIndex]
        frameSign = np.sign(frame)
        frameDiff = np.diff(frameSign) * 0.5
        return np.sum(np.abs(frameDiff)) / len(frameDiff)
