"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       callbacks.py
    Classes:    CollectionMethodCallbacks,
                TestTimeSeriesAnalysisFrames,
                TestFreqSeriesAnalysisFrames,
                TestFreqCenterOfMass,

    Author:     Landon Buell
    Date:       March 2024
"""

        #### IMPORTS ####

import analysisFrames
import collectionMethod

        #### CLASS DEFITIONS ####

class CollectionMethodCallbacks:
    """ 
        Static class of methods with signature:

        [bool] = callback([signalData.SignalData])

    """

    @staticmethod
    def signalHasAnalysisFramesTime(signalData) -> bool:
        """ Ensure that a provided signal has time-series analysis frames """
        return (signalData.cachedData.analysisFramesTime is not None)

    @staticmethod
    def signalHasAnalysisFramesFreq(signalData) -> bool:
        """ Ensure that a provided signal has freq-series analysis frames """
        return (signalData.cachedData.analysisFramesFreq is not None)

    @staticmethod
    def makeDefaultTimeSeriesAnalysisFrames(signalData) -> bool:
        """ Create the time-series analysis frames for the signal using the 'default' params """
        params = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        signalData.makeTimeSeriesAnalysisFrames(params)
        return (signalData.cachedData.analysisFramesTime is not None)

    @staticmethod
    def makeDefaultFreqSeriesAnalysisFrames(signalData) -> bool:
        """ Create the freq-series analysis frames for the signal using the 'default' params """
        params = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        signalData.makeFreqSeriesAnalysisFrames(params)
        return (signalData.cachedData.analysisFramesFreq is not None)

    @staticmethod
    def makeDefaultFreqCenterOfMasses(signalData) -> bool:
        """ Create the freq-series center of mass for each freq-series analysis frame """
        params = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        signalData.makeFrequencyCenterOfMass(params)
        return (signalData.cachedData.freqCenterOfMasses is not None)

        #### CLASSES FOR TESTING CALLBACKS ####

class TestTimeSeriesAnalysisFrames(collectionMethod.AbstractCollectionMethod):
    """ Class to test Signal Data Callbacks """

    __NAME = "TestTimeSeriesAnalysisFrames"

    def __init__(self):
        """ Constructor """
        super().__init__(TestTimeSeriesAnalysisFrames.__NAME,1)
        self._callbacks.append( CollectionMethodCallbacks.makeDefaultTimeSeriesAnalysisFrames )

    def __del__(self):
        """ Destructor """
        super().__del__()

    def _callBody(self,signalData) -> bool:
        """ OVERRIDE: main body of call function """
        return True

class TestFreqSeriesAnalysisFrames(collectionMethod.AbstractCollectionMethod):
    """ Class to test Signal Data Callbacks """

    __NAME = "TestFreqSeriesAnalysisFrames"

    def __init__(self):
        """ Constructor """
        super().__init__(TestFreqSeriesAnalysisFrames.__NAME,1)
        self._callbacks.append( CollectionMethodCallbacks.makeDefaultFreqSeriesAnalysisFrames )

    def __del__(self):
        """ Destructor """
        super().__del__()

    def _callBody(self,signalData) -> bool:
        """ OVERRIDE: main body of call function """
        return True

class TestFreqCenterOfMass(collectionMethod.AbstractCollectionMethod):
    """ Class to test Signal Data Callbacks """

    __NAME = "TestFreqCenterOfMass"

    def __init__(self):
        """ Constructor """
        super().__init__(TestFreqCenterOfMass.__NAME,1)
        self._callbacks.append( CollectionMethodCallbacks.makeDefaultFreqCenterOfMasses )

    def __del__(self):
        """ Destructor """
        super().__del__()

    def _callBody(self,signalData) -> bool:
        """ OVERRIDE: main body of call function """
        return True