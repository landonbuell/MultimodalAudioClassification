"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       unitTests.py
    Classes:    UnitTestCollectionMethods

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import numpy as np
import callbacks

import signalData
import dummyWaveforms

import timeDomainEnvelope
import zeroCrossingRate


        #### CLASS DEFINTIONS ####

class UnitTestCollectionMethods:
    """ Runs Unit Tests on provided signals """

    def __init__(self,
                 listOfMethods=None,
                 listOfWaveforms=None):
        """ Constructor """
        self._listOfMethods     = [] # List of feature collection methods
        self._listOfSignals     = [] # List of np arrays

        if (listOfMethods is not None):
            self._listOfMethods = listOfMethods[:]
        if (listOfWaveforms is not None):
            self._listOfSignals = listOfWaveforms[:]

    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    def runAll(self):
        """ Run all signals against all collection methods """
        allFeatureNames = self.__getAllFeatureNames()
        numFeatures = len(allFeatureNames)
        for ii,waveform in enumerate(self._listOfSignals):
            features    = np.zeros(shape=(numFeatures,),dtype=np.float32)
            signal      = signalData.SignalData(waveform=waveform)
            self.__evaluateCollectionMethods(signal,features)
        return None

    # Private Interface

    def __getAllFeatureNames(self) -> int:
        """ Return a list of all feature Names """
        featureNames = []
        for method in self._listOfMethods:
            featureNames += method.featureNames()
        return featureNames

    def __evaluateCollectionMethods(self,
                                    signal: signalData.SignalData,
                                    features: np.ndarray) -> None:
        """ Evaluate signal against collection methods """
        featureCounter = 0
        for ii,method in enumerate(self._listOfMethods):          
            success = method.call(signal)
            if (success == False):
                data = np.full(shape=(method.getNumFeatures(),),fill_value=np.NaN)
            else:
                data = method.getFeatures()
            # Add features
            for jj in range(data.size):
                features[featureCounter] = data[jj]
                featureCounter += 1
        # Done!
        return None

class PresetUnitTests:
    """ Static class of pre-existing unit test """

    @staticmethod
    def getTestBasicTimeSeriesMethods():
        """ Return test of basic time-series methods """
        methods = [timeDomainEnvelope.TimeDomainEnvelope(4),
                   zeroCrossingRate.TotalZeroCrossingRate(),]
        signals = [dummyWaveforms.getUniformWhiteNoise(),
                   dummyWaveforms.getSine440HzSignal(),]
        tests = UnitTestCollectionMethods(methods,signals)
        return tests

    @staticmethod
    def getTestCachedData():
        """ Return test for time-series analysis frames """
        methods = [callbacks.TestTimeSeriesAnalysisFrames(),
                   callbacks.TestFreqSeriesAnalysisFrames(),
                   callbacks.TestFreqCenterOfMass(),]
        signals = [ dummyWaveforms.getSine440Hz880HzSignal(),
                    dummyWaveforms.getLinearRampSignal(),]
        tests = UnitTestCollectionMethods(methods,signals)
        return tests

