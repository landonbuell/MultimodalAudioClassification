"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       melFilterBankEnergies.py
    Classes:    __MelFilterBankEnergies,
                MelFilterBankEnergyMeans
                MelFilterBankEnergyVariances
                MelFilterBankEnergyMedians
                MelFilterBankEnergiesMinMax

    Author:     Landon Buell
    Date:       June 2024
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt

import collectionMethod

import analysisFrames

        #### CLASS DEFINITIONS ####

class MelFilterBankEnergies(collectionMethod.AbstractCollectionMethod):
    """
        Abstract Base class for other methods that use ta signal's MFBE's for features
    """

    __NAME = "MelFilterBankEnergies"

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
                 numFilters: int,
                 forceRemake=False,
                 normalize=True):
        """ Constructor """
        super().__init__(MelFilterBankEnergies.__NAME,
                         numFilters * frameParams.maxNumFrames)
        self._params        = frameParams
        self._numFilters    = numFilters
        self._forceRemake   = forceRemake
        self._normalize     = normalize

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def numFilters(self) -> int:
        """ Return the number of filters """
        return self._numFilters

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute MFBE's for signal """
        if (self._makeMfbes(signal) == False):
            return False
        np.copyto(self._data,signal.cachedData.melFilterFrameEnergies.getEnergies())
        return True

    def _makeMfbes(self,
                   signal: collectionMethod.signalData.SignalData) -> bool:
        """ Make Mel Frequency Cepstrum Coefficients """
        madeMFBEs = signal.makeMelFilterBankEnergies(
            self.numFilters,self._params,self._forceRemake)
        if (madeMFBEs == False):
            msg = "Failed to make Mel Filter bank energies for signal: {0}".format(signal)
            self._logMessage(msg)
            return False
        return True

    def _plotEnergiesByFrame(self,
                              signal: collectionMethod.signalData.SignalData):
        """ Create a plot to show the the energy of each filter bank changes by each frame """
        plt.figure(figsize=(16,12),facecolor="gray")
        plt.title("Mel Filter Bank Energies by Frame",size=32,weight="bold")
        plt.xlabel("Frame Index",size=24,weight="bold")
        plt.ylabel("Energy Level",size=24,weight="bold")

        energies = np.transpose(signal.cachedData.melFilterFrameEnergies.getEnergies())
        for ii in range(self.numFilters):
            energyData = np.log10(energies[ii])
            label = "MFBE #{0}".format(ii)
            plt.plot(energyData,label=label)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return None

    # Static Interface

    @staticmethod
    def hertzToMels(hertz: np.ndarray) -> np.ndarray:
        """ Cast Hz to Mels """
        return 2595 * np.log10(1 + (hertz/700))

    @staticmethod
    def melsToHertz(mels: np.ndarray) -> np.ndarray:
        """ Cast Mels to Hz """
        return 700 * (10**(mels/2595) - 1)

class MelFilterBankEnergyMeans(MelFilterBankEnergies):
    """
        Compute + Return the Mean of each MFBE across all analysis frames
    """

    __NAME = "MelFilterBankEnergyMeans"

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
                 numFilters: int,
                 forceRemake=False,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,numFilters,forceRemake,normalize)
        self._name  = MelFilterBankEnergyMeans.__NAME
        self._data  = np.zeros(shape=(numFilters,),dtype=np.float32)

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute average MFBE's for signal """
        if (self._makeMfbes(signal) == False):
            return False
        np.copyto(self._data,signal.cachedData.melFilterFrameEnergies.getMeans(self._normalize))
        return True

class MelFilterBankEnergyVaris(MelFilterBankEnergies):
    """
        Compute + Return the Variance of each MFBE across all analysis frames
    """

    __NAME = "MelFilterBankEnergyVaris"

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
                 numFilters: int,
                 forceRemake=False,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,numFilters,forceRemake,normalize)
        self._data  = np.zeros(shape=(numFilters,),dtype=np.float32)
        self._name = MelFilterBankEnergyVaris.__NAME

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute MFBE's for signal """
        success = super()._callBody(signal)
        if (success == False):
            return False
        np.copyto(self._data,signal.cachedData.melFilterFrameEnergies.getVariances(self._normalize))
        return True

class MelFilterBankEnergyMedians(MelFilterBankEnergies):
    """
        Compute + Return the Medians of each MFBE across all analysis frames
    """

    __NAME = "MelFilterBankEnergyMedians"

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
                 numFilters: int,
                 forceRemake=False,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,numFilters,forceRemake,normalize)
        self._data  = np.zeros(shape=(numFilters,),dtype=np.float32)
        self._name  = MelFilterBankEnergyMedians.__NAME

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute MFBE's for signal """
        success = super()._callBody(signal)
        if (success == False):
            return False
        np.copyto(self._data,signal.cachedData.melFilterFrameEnergies.getMedians(self._normalize))
        return True

class MelFilterBankEnergyMinMax(MelFilterBankEnergies):
    """
        Compute + Return the Min & Max of each MFBE across all analysis frames
    """
    
    __NAME = "MelFilterBankEnergyMinMax"

    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
                 numFilters: int,
                 forceRemake=False,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,numFilters * 2,forceRemake,normalize)
        self._data  = np.zeros(shape=(numFilters,),dtype=np.float32)
        self._name = MelFilterBankEnergyMinMax.__NAME

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute MFBE's for signal """
        success = super()._callBody(signal)
        if (success == False):
            return False
        halfNumFeatures = int(self.getNumFeatures() // 2)
        np.copyto(self._data[:halfNumFeatures], signal.cachedData.melFilterFrameEnergies.getMins())
        np.copyto(self._data[halfNumFeatures:], signal.cachedData.melFilterFrameEnergies.getMaxes())
        return True
