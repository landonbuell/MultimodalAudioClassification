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
        self._methods = []

        self._ptrPipelineMgr = None
        
        self._callbacksPreprocessSignal     = []
        self._callbacksPreprocessFeatures   = []
        self._callbacksPostProcessSignal    = []
        self._callbacksPostProcessFeatures  = []
        
    def __del__(self):
        """ Destructor """
        self._methods.clear()
        self._ptrPipelineMgr = None

    @staticmethod
    def deepCopy(source,
                 identifier=None):
        """ Generate a deep copy of this instance """
        if (identifier is None):
            identifier = "copyOf" + source.getName()
        newPipeline = FeaturePipeline(identifier)
        # Copy Members
        newPipeline._methods        = source._methods[:] #Copy copy underlying list
        newPipeline._ptrPipelineMgr = source._ptrPipelineMgr
        newPipeline._callbacksPreprocessSignal     = source._callbacksPreprocessSignal[:]
        newPipeline._callbacksPreprocessFeatures   = source._callbacksPreprocessFeatures[:]
        newPipeline._callbacksPostProcessSignal    = source._callbacksPostProcessSignal[:]
        newPipeline._callbacksPostProcessFeatures  = source._callbacksPostProcessFeatures[:]
        return newPipeline

    # Accessors

    def getName(self) -> str:
        """ Return the name of this pipeline """
        return self._indentifier

    def getSize(self) -> int:
        """ Return the number of items in the pipeline """
        return len(self._methods)

    def getNumFeatures(self) -> int:
        """ Return the number of features in this pipeline """
        numFeatures = 0
        for method in self._methods:
            if (method is not None):
                numFeatures += method.getNumFeatures()
        return numFeatures

    def getManager(self):
        """ Return a pointer to the manager that owns this pipeline """
        return self._ptrPipelineMgr

    def setManager(self,pipelineMgr) -> None:
        """ Set the pointer to the manager that owns this pipeline """
        self._ptrPipelineMgr = pipelineMgr
        return None

    # Public Interface

    def appendCollectionMethod(self,
                               collectionMethod) -> None:
        """ Append a collection method to this pipeline """
        self._methods.append(collectionMethod)
        return None

    def evaluate(self,
                 signal: signalData.SignalData) -> featureVector.FeatureVector:
        """ Evaluate all collection methods and populate the feature vector """
        features = featureVector.FeatureVector(self.getNumFeatures(),signal.getTarget())
        self.__evaluateSignalPreprocessCallbacks(signal)
        self.__evaluateFeaturePreprocessCallbacks(features)
        self.__evaluateHelper(signal,features)
        self.__evaluateSignalPostprocessCallbacks(signal)
        self.__evaluateFeaturePostprocessCallbacks(features)
        return features

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


class DefaultFeaturePipeline:
    """ Static Class of Default Feature Pipelines """

    def getDefaultPipeline00() -> FeaturePipeline:
        """ Get the default pipeline 00 """
        pipeline = FeaturePipeline("Alpha")
        return pipeline

    def getDefaultPipeline01() -> FeaturePipeline:
        """ Get the default pipeline 01 """
        pipeline = FeaturePipeline("Beta")
        return pipeline

