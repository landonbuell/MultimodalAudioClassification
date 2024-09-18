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
        super().__init__(MelFrequencyCepstrumCoefficients.__NAME,
                         numCoeffs * frameParams.maxNumFrames)
        self._params        = frameParams
        self._numCoeffs     = numCoeffs
        self._forceRemake   = forceRemake
        self._normalize     = normalize

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def numCoeffs(self) -> int:
        """ Return the number of MFCC's """
        return self._numCoeffs

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute MFCC's for signal """
        if (self._makeMfccs(signal) == False):
            return False
        np.copyto(self._data,signal.cachedData.melFreqCepstralCoeffs.getCoeffs().flatten())
        return True

    def _makeMfccs(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute MFCC's for signal """
        madeMFCCs = signal.makeMelFrequencyCepstralCoeffs(
            self.numCoeffs,self._params,self._forceRemake)
        if (madeMFCCs == False):
            msg = "Failed to make Mel Frequency Cepstral Coefficients for signal: {0}".format(signal)
            self._logMessage(msg)
            return False
        return True

    def _plotCoefficientsByFrame(self,
                              signal: collectionMethod.signalData.SignalData):
        """ Create a plot to show the the energy of each filter bank changes by each frame """
        plt.figure(figsize=(16,12),facecolor="gray")
        plt.title("Mel Filter Bank Energies by Frame",size=32,weight="bold")
        plt.xlabel("Frame Index",size=24,weight="bold")
        plt.ylabel("Energy Level",size=24,weight="bold")

        energies = np.transpose(signal.cachedData.melFreqCepstralCoeffs.getCoeffs())
        for ii in range(self.numFilters):
            energyData = np.log10(energies[ii])
            label = "MFBE #{0}".format(ii)
            plt.plot(energyData,label=label)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return None

class MelFrequencyCepstrumCoefficientMeans(MelFrequencyCepstrumCoefficients):
    """
        Compute + Return the mean of eah MFCC across all analysis frames 
    """

    __NAME = "MelFrequencyCepstrumCoefficientMeans"

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
                 numCoeffs: int,
                 forceRemake=False,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,numCoeffs,forceRemake,normalize)
        self._name = MelFrequencyCepstrumCoefficientMeans.__NAME
        self._data = np.zeros(shape=(numCoeffs,),dtype=np.float32)

    def __del__(self):
        """ Destructor """
        pass

    # Protected Interface

    def _copyBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute average MFCC's for signal """
        success = super()._callBody(signal)
        if (success == False):
            return False
        np.copyto(self._data,signal.cachedData.melFreqCepstralCoeffs.getMeans(self._normalize))
        return True

class MelFrequencyCepstrumCoefficientVaris(MelFrequencyCepstrumCoefficients):
    """
        Compute + Return the varaiance of eah MFCC across all analysis frames 
    """

    __NAME = "MelFrequencyCepstrumCoefficientVaris"

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
                 numCoeffs: int,
                 forceRemake=False,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,numCoeffs,forceRemake,normalize)
        self._name = MelFrequencyCepstrumCoefficientVaris.__NAME
        self._data = np.zeros(shape=(numCoeffs,),dtype=np.float32)

    def __del__(self):
        """ Destructor """
        pass

    # Protected Interface

    def _copyBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute median MFCC's for signal """
        success = super()._callBody(signal)
        if (success == False):
            return False
        np.copyto(self._data,signal.cachedData.melFreqCepstralCoeffs.getVariances(self._normalize))
        return True

class MelFrequencyCepstrumCoefficientMedians(MelFrequencyCepstrumCoefficients):
    """
        Compute + Return the medians of eah MFCC across all analysis frames 
    """

    __NAME = "MelFrequencyCepstrumCoefficientMedians"

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
                 numCoeffs: int,
                 forceRemake=False,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,numCoeffs,forceRemake,normalize)
        self._name = MelFrequencyCepstrumCoefficientMedians.__NAME
        self._data = np.zeros(shape=(numCoeffs,),dtype=np.float32)

    def __del__(self):
        """ Destructor """
        pass

    # Protected Interface

    def _copyBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute median MFCC's for signal """
        success = super()._callBody(signal)
        if (success == False):
            return False
        np.copyto(self._data,signal.cachedData.melFreqCepstralCoeffs.getMedians(self._normalize))
        return True

class MelFrequencyCepstrumCoefficientMinMax(MelFrequencyCepstrumCoefficients):
    """
        Compute + Return the Min & Max of eah MFCC across all analysis frames 
    """

    __NAME = "MelFrequencyCepstrumCoefficientMinMax"

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
                 numCoeffs: int,
                 forceRemake=False,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,numCoeffs,forceRemake,normalize)
        self._name = MelFrequencyCepstrumCoefficientMinMax.__NAME
        self._data = np.zeros(shape=(numCoeffs,),dtype=np.float32)

    def __del__(self):
        """ Destructor """
        pass

    # Protected Interface

    def _copyBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute mins & maxs of MFCC's for signal """
        success = super()._callBody(signal)
        if (success == False):
            return False
        halfNumFeatures = int(self.getNumFeatures() // 2)
        np.copyto(self._data[:halfNumFeatures], signal.cachedData.melFreqCepstralCoeffs.getMins())
        np.copyto(self._data[halfNumFeatures:], signal.cachedData.melFreqCepstralCoeffs.getMaxes())
        return True
