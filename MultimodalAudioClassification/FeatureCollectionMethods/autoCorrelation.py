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
                 numCoeffs: int):
        """ Constructor """
        super().__init__(AutoCorrelationCoefficients.__NAME,
                         numCoeffs)

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
        sumUpperBound = signal.getNumSamples() - coeffIndex
        sums = np.zeros(shape=(3,),dtype=np.float32)
        for ii in range(sumUpperBound):
            sums[0] += (signal.waveform[ii] * signal.waveform[ii + coeffIndex])
            sums[1] += (signal.waveform[ii] * signal.waveform[ii])
            sums[2] += (signal.waveform[ii + coeffIndex] + signal.waveform[ii + coeffIndex])
        # Now Compute the result
        sums[1] = np.sqrt(sums[1])
        sums[2] = np.sqrt(sums[2])
        coeff = sums[0] / (sums[1] * sums[2])
        return coeff