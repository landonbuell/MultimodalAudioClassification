"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       featurePipeline.py
    Classes:    FeaturePipeline

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import signalData
import featureVector

        #### CLASS DEFINTIONS ####

class FeaturePipeline:
    """ Stores group of feature collection methods to be executed in sequence """

    __MAX_NUM_COLLECTION_METHODS = 64

    def __init__(self,
                 identifier: str):
        """ Constructor """
        self._indentifier = identifier
        self._ptrPipelineMgr = None

        self._methods = [None] * FeaturePipeline.__MAX_NUM_COLLECTION_METHODS
        self._size = 0

        self._callbacksPreprocessSignal     = []
        self._callbacksPreprocessFeatures   = []
        self._callbacksPostProcessSignal    = []
        self._callbacksPostProcessFeatures  = []
        
    def __del__(self):
        """ Destructor """
        self._methods.clear()

    # Accessors

    def getName(self) -> str:
        """ Return the name of this pipeline """
        return self._indentifier

    def getSize(self) -> int:
        """ Return the number of items in the pipeline """
        return len(self._size)

    def getNumFeatures(self) -> int:
        """ Return the number of features in this pipeline """
        numFeatures = 0
        for method in self._methods:
            if (method is not None):
                numFeatures += method.getNumFeatures()
        return numFeatures

    # Public Interface

    def appendCollectionMethod(self,
                               collectionMethod) -> None:
        """ Append a collection method to this pipeline """
        self._methods[self._size] = collectionMethod
        self._size += 1
        return None

    def evaluate(self,
                 signal: signalData.SignalData,
                 vector: featureVector.FeatureVector) -> None:
        """ Evaluate all collection methods and populate the feature vector """
        self.__evaluateSignalPreprocessCallbacks(signal)
        self.__evaluateFeaturePreprocessCallbacks(vector)
        self.__evaluateHelper(signal,vector)
        self.__evaluateSignalPostprocessCallbacks(signal)
        self.__evaluateFeaturePostprocessCallbacks(vector)
        return None

    # Private Interface

    def __evaluateHelper(self,
                            signal: signalData.SignalData,
                            vector: featureVector.FeatureVector) -> None:
        """ Helper function to evaluate the feature pipeline """
        featuresCollected = 0
        for method in self._methods:
            if (method is None):
                continue
            features = method.call(signal)
            if (len(features) != method.getNumFeatures()):
                msg = "Exepected collection method {0} to return {1} features but got {2}".format(
                    str(method),method.getNumFeatures(),len(features))
                raise RuntimeError(msg)
            for ii in range(method.getNumFeatures()):
                vector[featuresCollected] = features[ii]
                featuresCollected += 1
            # Completed with current method
        return None

    def __evaluateSignalPreprocessCallbacks(self,
                                            signal: signalData.SignalData) -> None:
        """ Evaluate all of the signal preprocessing callbacks """
        for callback in self._callbacksPreprocessSignal:
            callback.__call__(signal,self)
        return None

    def __evaluateFeaturePreprocessCallbacks(self,
                                             vector: featureVector.FeatureVector) -> None:
        """ Evaluate all of the signal preprocessing callbacks """
        for callback in self._callbacksPreprocessFeatures:
            callback.__call__(vector,self)
        return None

    def __evaluateSignalPostprocessCallbacks(self,
                                            signal: signalData.SignalData) -> None:
        """ Evaluate all of the signal preprocessing callbacks """
        for callback in self._callbacksPostprocessSignal:
            callback.__call__(signal,self)
        return None

    def __evaluateFeaturePostprocessCallbacks(self,
                                             vector: featureVector.FeatureVector) -> None:
        """ Evaluate all of the signal preprocessing callbacks """
        for callback in self._callbacksPostprocessFeatures:
            callback.__call__(vector,self)
        return None

    # Magic Methods

    def __len__(self):
        """ Return the number of items in the pipeline """
        return self._size

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))