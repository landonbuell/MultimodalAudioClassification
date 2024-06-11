"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       cepstralCoefficients.py
    Classes:    __MelFrequencyCepstrumCoefficients,
                MelFrequencyCepstrumCoefficientMeans
                MelFrequencyCepstrumCoefficientVariances
                MelFrequencyCepstrumCoefficientMedians
                MelFrequencyCepstrumCoefficientMinMax

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt

import analysisFrames

import collectionMethod

        #### CLASS DEFINITIONS ####

class MelFrequencyCepstrumCoefficients(collectionMethod.AbstractCollectionMethod):
    """
        Compute Mel Frequency Ceptral Coefficients
    """

    __NAME = "MelFrequencyCepstrumCoefficients"

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
                 numCoeffs: int,
                 forceRemake=False,
                 normalize=True):
        """ Constructor """
        super().__init__(MelFrequencyCepstrumCoefficients.__NAME,numCoeffs)
        self._params        = frameParams
        self._forceRemake   = forceRemake
        self._normalize     = normalize

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def numCoeffs(self) -> int:
        """ Return the number of MFCC's """
        return self._data.size

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute MFCC's for signal """
        madeMFCCs = signal.makeMelFrequencyCepstralCoeffs(
            self.numFilters,self._params,self._forceRemake)
        if (madeMFCCs == False):
            msg = "Failed to make Mel Frequency Cepstral Coefficients for signal: {0}".format(signal)
            self._logMessage(msg)
            return False
        return True
