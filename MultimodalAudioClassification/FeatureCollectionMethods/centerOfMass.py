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
        result = ["{0}{1}".format(self._name,str(self._weightType)),]
        return result

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: main body of call function """
        if (self._weightKernel.size != signal.waveform.size):
            self.__initWeightsKernel(signal.waveform.size)
        # Compute Center of Mass
        numerator   = np.dot(np.abs(signal.waveform),self._weightKernel)
        denominator = np.sum(np.abs(signal.waveform)) + collectionMethod.AbstractCollectionMethod.DELTA
        self._data[0] = numerator / denominator
        return True

    # Private Interface

    def __initWeightsKernel(self, newSize: int) -> None:
        """ Allocate a weights kernel of a new size """
        linearWeights = np.arange(newSize,dtype=np.float32)
        if (self._weightType == collectionMethod.WeightingFunction.LINEAR):
            self._weightKernel = linearWeights
        elif (self._weightType == collectionMethod.WeightingFunction.QUADRATIC):
            self._weightKernel = linearWeights**2
        elif (self._weightKernel == collectionMethod.WeightingFunction.LOG_NATURAL):
            self._weightKernel = np.log(linearWeights + collectionMethod.AbstractCollectionMethod.DELTA) 
        elif (self._weightKernel == collectionMethod.WeightingFunction.LOG_BASE10):
            self._weightKernel = np.log10(linearWeights + collectionMethod.AbstractCollectionMethod.DELTA) 
        else:
            msg = "Unrecognized weight type: {0}. Defaulting to linear"
            self._weightKernel = collectionMethod.WeightingFunction.LINEAR
            raise RuntimeWarning(msg)
        return None

class FrequencyCenterOfMass(collectionMethod.AbstractCollectionMethod):
    """
        Compute the frequency center-of-mass for eacg analysis frame
    """

    __NAME = "FrequencyCenterOfMass"


    def __init__(self,
                 useAverage=True,
                 useVariance=True,
                 useDiffMinMax=True):
        """ Constructor """
        super().__init__(TemporalCenterOfMass.__NAME,1)
        self._uses = [useAverage,useVariance,useDiffMinMax,]
        if (sum(self._uses) == 0):
            msg = "Must use some information from FrequencyCenterOfMass"
            raise RuntimeError(msg)
        self._data = np.zeros(shape=(sum(self._uses),),dtype=np.float32)
        self._callbacks.append( collectionMethod.CollectionMethodCallbacks.makeDefaultFreqCenterOfMasses )

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def useAverage(self) -> bool:
        """ Return T/F if we should use the average FCOM """
        return self._uses[0]

    @property
    def useVariance(self) -> bool:
        """ Return T/F if we should use the variance of the FCOM """
        return self._uses[1]

    @property
    def useDiffMinMax(self) -> bool:
        """ Return T/F if we should use the diff of the min/max FCOM """
        return self._uses[2]
   
    # Public Interface

    def featureNames(self) -> list:
        """ OVERRIDE: Return a list of the feature names """
        result = []
        # TODO: THIS!
        return None

    # Protected Interface

    def _callBody(self,
                  signal: collectionMethod.signalData.SignalData) -> bool:
        """ OVERRIDE: main body of call function """
        
        return True

    # Private Interface

    def __initWeightsKernel(self, newSize: int) -> None:
        """ Allocate a weights kernel of a new size """
        linearWeights = np.arange(newSize,dtype=np.float32)
        if (self._weightType == collectionMethod.WeightingFunction.LINEAR):
            self._weightKernel = linearWeights
        elif (self._weightType == collectionMethod.WeightingFunction.QUADRATIC):
            self._weightKernel = linearWeights**2
        elif (self._weightKernel == collectionMethod.WeightingFunction.LOG_NATURAL):
            self._weightKernel = np.log(linearWeights + collectionMethod.AbstractCollectionMethod.DELTA) 
        elif (self._weightKernel == collectionMethod.WeightingFunction.LOG_BASE10):
            self._weightKernel = np.log10(linearWeights + collectionMethod.AbstractCollectionMethod.DELTA) 
        else:
            msg = "Unrecognized weight type: {0}. Defaulting to linear"
            self._weightKernel = collectionMethod.WeightingFunction.LINEAR
            raise RuntimeWarning(msg)
        return None