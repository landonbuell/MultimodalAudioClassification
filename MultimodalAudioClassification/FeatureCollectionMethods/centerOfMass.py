"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       centerOfMass.py
    Classes:    TemporalCenterOfMass,
                FrequencyCenterOfMass,
    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import numpy as np

import collectionMethod

        #### CLASS DEFINITIONS ####

class TemporalCenterOfMass(collectionMethod.AbstractCollectionMethod):
    """
        Compute the temporal center-of-mass for the waveform
    """

    __NAME = "TemporalCenterOfMass"
    __NUM_FEATURES = 1

    def __init__(self,
                 weightingFunctionType: collectionMethod.WeightingFunction):
        """ Constructor """
        super().__init__(TemporalCenterOfMass.__NAME,
                         TemporalCenterOfMass.__NUM_FEATURES)
        self._weightType    = weightingFunctionType
        self._weightKernel  = np.zeros(shape=(1,),dtype=np.float32) 

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    # Public Interface

    def featureNames(self) -> list:
        """ OVERRIDE: Return a list of the feature names """
        result = ["{0}.{1}".format(self._name,str(self._weightType)),]
        return result

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData,
                  features: collectionMethod.featureVector.FeatureVector) -> bool:
        """ OVERRIDE: main body of call function """
        if (self._weightKernel.size != signal.waveform.size):
            self.__initWeightsKernel(signal.waveform.size)
        # Compute Center of Mass
        numerator   = np.dot(np.abs(signal.waveform),self._weightKernel)
        denominator = np.sum(np.abs(signal.waveform)) + collectionMethod.AbstractCollectionMethod.DELTA
        features.appendItem( numerator / denominator )
        return True

    # Private Interface

    def __initWeightsKernel(self, newSize: int) -> None:
        """ Allocate a weights kernel of a new size """
        linearWeights = np.arange(newSize,dtype=np.float32)
        if (self._weightType == collectionMethod.WeightingFunction.LINEAR):
            self._weightKernel = linearWeights
        elif (self._weightType == collectionMethod.WeightingFunction.QUADRATIC):
            self._weightKernel = linearWeights * linearWeights
        elif (self._weightType == collectionMethod.WeightingFunction.LOG_NATURAL):
            self._weightKernel = np.log(linearWeights + collectionMethod.AbstractCollectionMethod.DELTA) 
        elif (self._weightType == collectionMethod.WeightingFunction.LOG_BASE10):
            self._weightKernel = np.log10(linearWeights + collectionMethod.AbstractCollectionMethod.DELTA) 
        else:
            msg = "Unrecognized weight type: {0}. Defaulting to linear"
            self._weightKernel = collectionMethod.WeightingFunction.LINEAR
            raise RuntimeWarning(msg)
        return None

class FrequencyCenterOfMass(collectionMethod.AbstractCollectionMethod):
    """
        Compute the frequency center-of-mass for each analysis frame
    """

    __NAME = "FrequencyCenterOfMass"
    __NUM_FEATURES = 6

    def __init__(self):
        """ Constructor """
        super().__init__(FrequencyCenterOfMass.__NAME,
                         FrequencyCenterOfMass.__NUM_FEATURES)
        self._callbacks.append( collectionMethod.CollectionMethodCallbacks.makeDefaultFreqCenterOfMasses )

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Public Interface

    def featureNames(self) -> list:
        """ OVERRIDE: Return a list of the feature names """
        result = [self.getName() + "Mean",
                  self.getName() + "Variance",
                  self.getName() + "Median",
                  self.getName() + "Min",
                  self.getName() + "Max",
                  self.getName() + "DiffMinMax",]
        return result

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData,
                  features: collectionMethod.featureVector.FeatureVector) -> bool:
        """ OVERRIDE: main body of call function """
        features.appendItem( np.mean(signal.cachedData.freqCenterOfMasses) )
        features.appendItem( np.var(signal.cachedData.freqCenterOfMasses) )
        features.appendItem( np.median(signal.cachedData.freqCenterOfMasses) )
        features.appendItem( np.min(signal.cachedData.freqCenterOfMasses) )
        features.appendItem( np.max(signal.cachedData.freqCenterOfMasses) )
        features.appendItem( (np.max(signal.cachedData.freqCenterOfMasses) - \
                           np.min(signal.cachedData.freqCenterOfMasses)) )
        return True

    # Private Interface
