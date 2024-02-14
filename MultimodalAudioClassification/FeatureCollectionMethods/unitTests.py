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

import signalData

import collectionMethod
import timeDomainEnvelope
import zeroCrossingRate
import temporalCenterOfMass
import autoCorrelation

        #### CLASS DEFINTIONS ####

class UnitTestCollectionMethods:
    """ Runs Unit Tests on provided signals """

    def __init__(self,
                 listOfMethods=None,
                 listOfSignals=None):
        """ Constructor """
        self._listOfMethods     = []
        self._listOfSignals     = []

        if (listOfMethods is not None):
            self._listOfMethods = listOfMethods[:]
        if (listOfSignals is not None):
            self._listOfSignals = listOfSignals[:]

    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    def registerCollectionMethod(self,
                                 collectionMethod) -> None:
        """ Add a collection method to the list to run """
        self._listOfMethods.append(collectionMethod)
        return None

    def registerSignalData(self,
                           signal) -> None:
        """ Add a signal to the list to run """
        self._listOfSignals.append(signal)
        return None

    def runAll(self):
        """ Run all signals against all collection methods """
        allFeatureNames = self.__getAllFeatureNames()
        numFeatures = len(allFeatureNames)
        for ii,signal in enumerate(self._listOfSignals):
            features = np.zeros(shape=(numFeatures,),dtype=np.float32)
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
                                    signal,
                                    features) -> None:
        """ Evaluate signal against collection methods """
        featureCounter = 0
        for ii,method in enumerate(self._listOfMethods):
            success = method.call(signal)
            if (success == False):
                data = np.full(shape=(method.getNumFeatures,),fill_value=np.NaN)
            else:
                data = method.getFeatures()
            # Add features
            for jj in range(len(data.size)):
                features[featureCounter] = data[jj]
                featureCounter += 1
        # Done!
        return None

class DummySignals:
    """ Static class of dummy signal getters """

    @staticmethod
    def getSine440HzSignal():
        """ Return Signal w/ 440Hz Sine waveform """
        t = np.arange(88200,dtype=np.float32)
        waveform = np.sin(2*np.pi*t*440)
        signal = signalData.SignalData()
        signal.setWaveform(waveform)
        return signal

    @staticmethod
    def getSine880HzSignal():
        """ Return Signal w/ 880 Hz Sine waveform """
        t = np.arange(88200,dtype=np.float32)
        waveform = np.sin(2*np.pi*t*880)
        signal = signalData.SignalData()
        signal.setWaveform(waveform)
        return signal

    @staticmethod
    def getNormalWhiteNoise():
        """ Return Signal w/ normalized white noise waveform """
        waveform = np.random.random(size=88200)
        waveform /= np.max(np.abs(waveform))
        signal = signalData.SignalData()
        signal.setWaveform(waveform)
        return signal

    @staticmethod
    def getConstZeroSignal():
        """ Return Signal w/ all zero waveform """
        waveform = np.zeros(shape=(88200,),dtype=np.float32)
        signal = signalData.SignalData()
        signal.setWaveform(waveform)
        return signal

    @staticmethod
    def getConstOneSignal():
        """ Return Signal w/ all 1's waveform """
        waveform = np.zeros(shape=(88200,),dtype=np.float32) + 1
        signal = signalData.SignalData()
        signal.setWaveform(waveform)
        return signal

    @staticmethod
    def getLinearRampSignal():
        """ Return Signal w/ increasing waveform """
        waveform = np.arange(88200,dtype=np.float32)
        signal = signalData.SignalData()
        signal.setWaveform(waveform)
        return signal

class PresetUnitTests:
    """ Static class of pre-existing unit test """

    @staticmethod
    def getTestBasicTimeSeriesMethods():
        """ Return test of basic time-series methods """
        methods = [timeDomainEnvelope.TimeDomainEnvelope(4),]
        signals = [DummySignals.getLinearRampSignal(),]
        tests = UnitTestCollectionMethods(methods,signals)
        return tests


