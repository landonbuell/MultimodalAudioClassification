"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       spectrogram.py
    Classes:    Spectrogram,

    Author:     Landon Buell
    Date:       February 2024
"""


        #### IMPORTS ####

import numpy as np

import collectionMethod
import analysisFrames

        #### CLASS DEFINITIONS ####

class Spectrogram(collectionMethod.AbstractCollectionMethod):
    """ 
        Compute and Store the spectrogram of a signal
    """

    __NAME = "SPECTROGRAM"
    
    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters):
        """ Constructor """
        super().__init__(Spectrogram.__NAME,frameParams.getFreqFramesNumFeatures())
        self._params = frameParams
        self._callbacks.append( collectionMethod.CollectionMethodCallbacks.signalHasAnalysisFramesTime )

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def numFrames(self) -> int:
        """ Return the max allowed number of frames """
        return self._params.maxNumFrames

    @property
    def frameSize(self) -> int:
        """ Return the size of each frame """
        return self._parmas.freqFrameSize

    # Protected Interface

    def _callBody(self, 
                  signal: signalData.SignalData) -> bool:
        """ OVERRIDE: main body of call function """
        signal.makeFreqSeriesAnalysisFrames(self._params)
        self._data = signal.cachedData.analysisFramesFreq.flatten()
        return True
    