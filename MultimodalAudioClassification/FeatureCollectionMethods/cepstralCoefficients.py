"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       collectionMethod.py
    Classes:    AbstractCollectionMethod,
                CollectionMethodCallbacks

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import numpy as np

import collectionMethod

        #### CLASS DEFINITIONS ####

class MelFilterBankEnergies:
    """
        Compute the MFBE's for a signal
    """

    @staticmethod
    def hertzToMels(hertz: np.ndarray) -> np.ndarray:
        """ Cast Hz to Mels """
        return 2595 * np.log10(1 + (hertz/700))

    @staticmethod
    def melsToHertz(mels: np.ndarray) -> np.ndarray:
        """ Cast Mels to Hz """
        return 700 * (np.power(10,(mels/2595)) - 1)

class MelFrequencyCepstrumCoefficients(collectionMethod.AbstractCollectionMethod):
    """
        Compute Mel Frequency Ceptral Coefficients
    """

    __NAME = "MelFrequencyCepstrumCoefficients"

    def __init__(self,
                 numCoeffs: int):
        """ Constructor """
        super().__init__(MelFrequencyCepstrumCoefficients.__NAME,numCoeffs)

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

        return True
