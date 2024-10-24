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
                 onlyFramesInUse=False,
                 normalize=True):
        """ Constructor """
        super().__init__(MelFrequencyCepstrumCoefficients.__NAME,
                         numCoeffs * frameParams.maxNumFrames)
        self._params        = frameParams
        self._numCoeffs     = numCoeffs
        self._forceRemake   = forceRemake
        self._framesInUse   = onlyFramesInUse
        self._normalize     = normalize

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def numCoeffs(self) -> int:
        """ Return the number of MFCC's """
        return self._numCoeffs

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
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute MFCC's for signal """
        if (self._makeMfccs(signal) == False):
            return False
        coeffs = signal.cachedData.melFreqCepstralCoeffs.getCoeffs()
        if (self.normalize == True):
            coeffs /= np.max(coeffs)
        np.copyto(self._data,coeffs.ravel())
        return True

    def _makeMfccs(self,
                   signal: collectionMethod.signalData.SignalData) -> bool:
        """ Compute MFCC's for signal """
        signal.makeMelFrequencyCepstralCoeffs(self.numCoeffs,self._params,self._forceRemake)
        if (signal.cachedData.melFreqCepstralCoeffs == None):
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
                 onlyFramesInUse=True,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,
                         numCoeffs,
                         forceRemake,
                         onlyFramesInUse,
                         normalize)
        self._name = MelFrequencyCepstrumCoefficientMeans.__NAME
        self._resizeData(numCoeffs)

        intendedShape = [self._numCoeffs,]
        self._setIntendedShape(intendedShape)

    def __del__(self):
        """ Destructor """
        pass

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute average MFCC's for signal """
        if (self._makeMfccs(signal) == False):
            return False
        meanCoeffs = signal.cachedData.melFreqCepstralCoeffs.getMeans(
            self.onlyIncludeFramesInUse,self.normalize)
        np.copyto(self._data,meanCoeffs)
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
                 onlyFramesInUse=True,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,
                         numCoeffs,
                         forceRemake,
                         onlyFramesInUse,
                         normalize)
        self._name = MelFrequencyCepstrumCoefficientVaris.__NAME
        self._resizeData(numCoeffs)

        intendedShape = [self._numCoeffs,]
        self._setIntendedShape(intendedShape)
     
    def __del__(self):
        """ Destructor """
        pass

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute median MFCC's for signal """
        if (self._makeMfccs(signal) == False):
            return False
        variCoeffs = signal.cachedData.melFreqCepstralCoeffs.getVariances(
            self.onlyIncludeFramesInUse,self.normalize)
        np.copyto(self._data,variCoeffs)
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
                 onlyFramesInUse=True,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,
                         numCoeffs,
                         forceRemake,
                         onlyFramesInUse,
                         normalize)
        self._name = MelFrequencyCepstrumCoefficientMedians.__NAME
        self._resizeData(numCoeffs)

        intendedShape = [self._numCoeffs,]
        self._setIntendedShape(intendedShape)

    def __del__(self):
        """ Destructor """
        pass

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute median MFCC's for signal """
        if (self._makeMfccs(signal) == False):
            return False
        mediCoeffs = signal.cachedData.melFreqCepstralCoeffs.getMedians(
            self.onlyIncludeFramesInUse,self.normalize)
        np.copyto(self._data,mediCoeffs)
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
                 onlyFramesInUse=True,
                 normalize=True):
        """ Constructor """
        super().__init__(frameParams,
                         numCoeffs,
                         forceRemake,
                         onlyFramesInUse,
                         normalize)
        self._name = MelFrequencyCepstrumCoefficientMinMax.__NAME
        self._resizeData(numCoeffs * 2)

        intendedShape = [self._numCoeffs * 2,]
        self._setIntendedShape(intendedShape)

    def __del__(self):
        """ Destructor """
        pass

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: Compute mins & maxs of MFCC's for signal """
        if (self._makeMfccs(signal) == False):
            return False
        halfNumFeatures = int(self.getNumFeatures() // 2)

        minCoeffs = signal.cachedData.melFreqCepstralCoeffs.getMins(
            self.onlyIncludeFramesInUse,self.normalize)
        np.copyto(self._data[:halfNumFeatures], minCoeffs)
            
        maxCoeffs = signal.cachedData.melFreqCepstralCoeffs.getMaxes(
            self.onlyIncludeFramesInUse,self.normalize)
        np.copyto(self._data[:halfNumFeatures], maxCoeffs)
        return True
