"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       collectionMethod.py
    Classes:    AbstractCollectionMethod

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import numpy as np

import signalData

import collectionMethod

        #### CLASS DEFINITIONS ####

class TimeDomainEnvelope(collectionMethod.AbstractCollectionMethod):
    """
        Divide a waveform into N segments and compute the energy in each
    """

    __NAME = "TimeDomainEnvelope"

    def __init__(self,
                 numPartitions: int):
        """ Constructor """
        super().__init__(TimeDomainEnvelope.__NAME,numPartitions)

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def numPartitions(self) -> int:
        """ Return the number of partitions """
        return self._data.size

    # Protected Interface

    def _callBody(self,
                  signal: signalData.SignalData):
        """ OVERRIDE: Compute TDE's for signal """
        partitionSize = np.floor(signal.getNumSamples / self.numPartitions)
        startIndex = 0
        for ii in range(self.numPartitions):
            for jj in range(startIndex,startIndex + partitionSize):
                self._data[ii] += (signal.waveform[jj] * signal.waveform[jj])
            self._data[ii] = np.sqrt(self._data[ii] / partitionSize)
            startIndex += partitionSize
        return True
