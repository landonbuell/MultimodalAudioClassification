"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       autoCorrelation.py
    Classes:    AutoCorrelationCoefficients,

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import numpy as np

import collectionMethod

        #### CLASS DEFINITIONS ####

class AutoCorrelationCoefficients(collectionMethod.AbstractCollectionMethod):
    """
        Compute the temporal center-of-mass for the waveform
    """

    __NAME = "AutoCorrelationCoefficients"

    def __init__(self,
                 numCoeffs: int,
                 coeffStepSize=1):
        """ Constructor """
        super().__init__(AutoCorrelationCoefficients.__NAME,
                         numCoeffs)
        self._stepSize = coeffStepSize
        self._sums = np.zeros(shape=(3,),dtype=np.float32)

    def __del__(self):
        """ Destructor """
        super().__del__()


    # Accessors

    @property
    def numCoeffs(self) -> int:
        """ Return the number of auto correlation coeffs to compute """
        return self._data.size

    # Protected Interface

    def _callBody(self, 
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: main body of call function """
        for ii in range(self.numCoeffs):
            self._data[ii] = self.__computeCoefficient(signal,ii)
        return True

    def __computeCoefficient(self,
                             signal: collectionMethod.signalData.SignalData,
                             coeffIndex: int) -> np.float32:
        """ Compute the Coeff at the provided index """
        coeffScaled      = (coeffIndex * self._stepSize)
        sliceSize       = signal.getNumSamples() - coeffScaled
        waveformFirstK  = signal.waveform[0 : sliceSize]
        waveformLastK   = signal.waveform[coeffScaled : coeffScaled + sliceSize] 

        # Compute Sums
        self._sums[0] = np.dot(waveformFirstK,waveformLastK)
        self._sums[1] = np.dot(waveformFirstK,waveformFirstK)
        self._sums[2] = np.dot(waveformLastK,waveformLastK)

        # Now Compute the result
        self._sums[1] = np.sqrt(self._sums[1])
        self._sums[2] = np.sqrt(self._sums[2])
        coeff = self._sums[0] / (self._sums[1] * self._sums[2])
        return coeff