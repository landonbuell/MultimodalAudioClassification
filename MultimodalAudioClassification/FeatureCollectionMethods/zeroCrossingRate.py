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

    def __init__(self):
        """ Constructor """
        super().__init__(FrameZeroCrossingRate.__NAME,
                         FrameZeroCrossingRate.__NUM_FEATURES)

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
        zxrs = np.empty(shape=(signal.cachedData.analysisFramesTime.maxNumFrames,),dtype=np.float32)
        for ii in range(signal.cachedData.analysisFramesTime.maxNumFrames):
            zxrs[ii] = self.__computeZeroCrossingRateOfFrame(signal,ii)
        # Store values
        features.appendItem( np.mean(zxrs) )
        features.appendItem( np.var(zxrs) )
        features.appendItem( np.median(zxrs) )
        features.appendItem( np.min(zxrs) )
        features.appendItem( np.max(zxrs) )
        features.appendItem( np.max(zxrs) - np.min(zxrs) )
        return True

    # Private Interface

    def __computeZeroCrossingRateOfFrame(self,
                                         signal: collectionMethod.signalData.SignalData,
                                         frameIndex: int) -> None:
        """ Compute the zero crossing rate for a chosen frame """
        frameSign = np.sign(signal.cachedData.analysisFramesTime[frameIndex])
        frameDiff = np.diff(frameSign) * 0.5
        return np.sum(frameDiff) / signal.frameSign.size
