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

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def numCoeffs(self) -> int:
        """ Return the number of auto correlation coeffs to compute """
        return self._numFeatures

    # Protected Interface

    def _callBody(self, 
                  signal: collectionMethod.signalData.SignalData,
                  features: collectionMethod.featureVector.FeatureVector) -> bool:
        """ OVERRIDE: main body of call function """
        for ii in range(self.numCoeffs):
            features.appendItem( self.__computeCoefficient(signal,ii) )
        return True

    def __computeCoefficient(self,
                             signal: collectionMethod.signalData.SignalData,
                             coeffIndex: int) -> np.float32:
        """ Compute the Coeff at the provided index """
        coeffScaled      = (coeffIndex * self._stepSize)
        sliceSize       = signal.getNumSamples() - coeffScaled
        waveformFirstK  = signal.waveform[0 : sliceSize]
        waveformLastK   = signal.waveform[coeffScaled : coeffScaled + sliceSize] 
        cachedSums = np.empty(shape=(3,),dtype=np.float32)

        # Compute Sums
        cachedSums[0] = np.dot(waveformFirstK,waveformLastK)
        cachedSums[1] = np.dot(waveformFirstK,waveformFirstK)
        cachedSums[2] = np.dot(waveformLastK,waveformLastK)

        # Now Compute the result
        cachedSums[1] = np.sqrt(cachedSums[1])
        cachedSums[2] = np.sqrt(cachedSums[2])
        coeff = cachedSums[0] / (cachedSums[1] * cachedSums[2])
        return coeff