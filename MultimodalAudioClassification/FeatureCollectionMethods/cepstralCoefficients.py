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
import matplotlib.pyplot as plt

import collectionMethod

import analysisFrames

        #### CLASS DEFINITIONS ####

class MelFilterBankEnergies(collectionMethod.AbstractCollectionMethod):
    """
        Compute the MFBE's for a signal
    """

    __NAME = "MelFilterBankEnergies"

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
                 numCoeffs: int):
        """ Constructor """
        super().__init__(MelFilterBankEnergies.__NAME,numCoeffs)
        self._params = frameParams

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def numFilters(self) -> int:
        """ Return the number of filters """
        return self._data.size

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute MFCC's for signal """
        filterBanks = MelFilterBankEnergies.createMelFilters(
            self._params.freqLowBoundHz,self._params.freqHighBoundHz,
            self.numFilters,44100,self._params.getFreqFrameSizeMasked())
        return True

    # Static Interface

    @staticmethod
    def hertzToMels(hertz: np.ndarray) -> np.ndarray:
        """ Cast Hz to Mels """
        return 2595 * np.log10(1 + (hertz/700))

    @staticmethod
    def melsToHertz(mels: np.ndarray) -> np.ndarray:
        """ Cast Mels to Hz """
        return 700 * (10**(mels/2595) - 1)

    @staticmethod
    def createMelFilters(  lowerFreqHz: float,
                            upperFreqHz: float,
                            numFilters: int,
                            sampleRate: int,
                            frameSize: int) -> np.ndarray:
        """ Create Mel Filter Banks """
        lowerFreqMels = MelFilterBankEnergies.hertzToMels(lowerFreqHz)
        upperFreqMels = MelFilterBankEnergies.hertzToMels(upperFreqHz)
        melPoints = np.linspace(lowerFreqMels,upperFreqMels,numFilters + 2)
        hzPoints = MelFilterBankEnergies.melsToHertz(melPoints)

        bins = np.floor((frameSize + 1) * hzPoints / sampleRate )
        filterBanks = np.zeros(shape=(numFilters,frameSize),dtype=np.float32)

        for ii in range(1, numFilters + 1, 1): 
            # Each filter
            freqLeft    = int(bins[ii - 1])
            freqRight   = int(bins[ii + 1])
            freqCenter  = int(bins[ii])

            for jj in range(freqLeft,freqCenter):
                filterBanks[ii-1,jj] = (jj - bins[ii-1]) / (bins[ii] - bins[ii - 1])
            for jj in range(freqCenter,freqRight):
                filterBanks[ii-1,jj] = (bins[ii+1] - jj) / (bins[ii + 1] - bins[ii])
        
        freqAxis = np.linspace(lowerFreqHz,upperFreqHz,frameSize)
        MelFilterBankEnergies.plotFilters(filterBanks,freqAxis)
        return filterbanks

    @staticmethod
    def plotFilters(filterMatrix: np.ndarray, freqAxis: np.ndarray) -> None:
        """ Plot all filters """
        plt.figure(figsize=(16,12))
        plt.title("Mel Filters",size=24,weight='bold')
        plt.xlabel("Frequency",size=20,weight='bold')
        plt.ylabel("Filter Strength",size=20,weight='bold')

        # Plot the Stuff
        numFilters = filterMatrix.shape[0]
        for ii in range(numFilters):
            plt.plot(freqAxis,filterMatrix[ii],label="Filter{0}".format(ii))

        # House Keeping
        plt.grid()
        plt.legend()
        plt.show()
        return None

class MelFrequencyCepstrumCoefficients(collectionMethod.AbstractCollectionMethod):
    """
        Compute Mel Frequency Ceptral Coefficients
    """

    __NAME = "MelFrequencyCepstrumCoefficients"

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
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
