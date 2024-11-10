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
                 onlyFramesInUse=False,
                 normalize=True):
        """ Constructor """
        super().__init__(MelFilterBankEnergies.__NAME,
                         numFilters * frameParams.maxNumFrames)
        self._params        = frameParams
        self._numFilters    = numFilters
        self._forceRemake   = forceRemake
        self._framesInUse   = onlyFramesInUse
        self._normalize     = normalize

        intendedShape = [self._params.maxNumFrames, self._numFilters,]
        self._setIntendedShape(intendedShape)

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def numFilters(self) -> int:
        """ Return the number of filters """
        return self._numFilters

    @property
    def onlyIncludeFramesInUse(self) -> bool:
        """ Return T/F if features should only include frames in use """
        return self._framesInUse

    @property
    def normalize(self) -> bool:
        """ Return T/F if return features should be normalized within +/- 1 """
        return self._normalize

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData,
                  features: collectionMethod.featureVector.FeatureVector) -> bool:
        """ OVERRIDE: Compute MFBE's for signal """
        if (self._makeMfbes(signal) == False):
            return False
        energies = signal.cachedData.melFilterFrameEnergies.getEnergies()
        if (self.normalize == True):
            energies /= np.max(energies)
        features.appendItems( energies.ravel() )
        return True

    def _makeMfbes(self,
                   signal: collectionMethod.signalData.SignalData) -> bool:
        """ Make Mel Frequency Cepstrum Coefficients """
        signal.makeMelFilterBankEnergies(self.numFilters,self._params,self._forceRemake)
        if (signal.cachedData.melFilterFrameEnergies == None):
            msg = "Failed to make Mel Filter Bank Energies for signal: {0}".format(signal)
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
                 onlyFramesInUse=True,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,
                         numFilters,
                         forceRemake,
                         onlyFramesInUse,
                         normalize)
        self._name  = MelFilterBankEnergyMeans.__NAME
        self._resizeData(numFilters)

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute average MFBE's for signal """
        if (self._makeMfbes(signal) == False):
            return False
        meanEnergies = signal.cachedData.melFilterFrameEnergies.getMeans(
            self.onlyIncludeFramesInUse,self.normalize)
        np.copyto(self._data,meanEnergies)
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
                 onlyFramesInUse=True,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,
                         numFilters,
                         forceRemake,
                         onlyFramesInUse,
                         normalize)
        self._name = MelFilterBankEnergyVaris.__NAME
        self._resizeData(numFilters)

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData,
                  features: collectionMethod.featureVector.FeatureVector) -> bool:
        """ OVERRIDE: Compute MFBE's for signal """
        if (self._makeMfbes(signal) == False):
            return False
        variEnergies = signal.cachedData.melFilterFrameEnergies.getVariances(
            self.onlyIncludeFramesInUse,self.normalize)
        features.appendItems(variEnergies)       
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
                 onlyFramesInUse=True,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,
                         numFilters,
                         forceRemake,
                         onlyFramesInUse,
                         normalize)
        self._name  = MelFilterBankEnergyMedians.__NAME
        self._resizeData(numFilters)

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData,
                  features: collectionMethod.featureVector.FeatureVector) -> bool:
        """ OVERRIDE: Compute MFBE's for signal """
        if (self._makeMfbes(signal) == False):
            return False
        mediEnergies = signal.cachedData.melFilterFrameEnergies.getMedians(
            self.onlyIncludeFramesInUse,self.normalize)
        features.appendItems( mediEnergies )    
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
                 onlyFramesInUse=True,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,
                         numFilters,
                         forceRemake,
                         onlyFramesInUse,
                         normalize)
        self._name = MelFilterBankEnergyMinMax.__NAME
        self._resizeData(numFilters * 2)

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData,
                  features: collectionMethod.featureVector.FeatureVector) -> bool:
        """ OVERRIDE: Compute MFBE's for signal """
        if (self._makeMfbes(signal) == False):
            return False
        #halfNumFeatures = int(self.getNumFeatures() // 2)

        minEngergies = signal.cachedData.melFilterFrameEnergies.getMins(
            self.onlyIncludeFramesInUse,self.normalize)
        features.appendItems( minEngergies )

        maxEnergies = signal.cachedData.melFilterFrameEnergies.getMaxes(
            self.onlyIncludeFramesInUse,self.normalize)
        features.appendItems( maxEnergies )
        return True
