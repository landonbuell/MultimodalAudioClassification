"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureCollection  
File:           CollectionMethods.py
 
Author:         Landon Buell
Date:           December 2021
"""

            #### IMPORTS ####

import os
import sys
from typing import Collection
import numpy as np
import scipy.fftpack as fftpack

import Administrative

EPSILON = np.array([1e-12],dtype=np.float32)

            #### CLASS DEFINIIONS ####

class CollectionMethod:
    """
    Abstract Base Class for All Collection Methods to Be Queued
    """

    def __init__(self,name,param):
        """ Constructor for CollectionMethod Base Class """
        self._methodName    = name
        self._parameter     = param
        self._owner         = None
        self._result        = np.empty(shape=(param,),dtype=np.float32)
        self._preprocessCallbacks = []
        self._postprocessCallbacks = []

        # Register a callback to log Execution
        #self.registerPreprocessCallback( CollectionMethod.logExecutionTimestamp )
       
    def __del__(self):
        """ Destructor for CollectionMethod Base Class """
        self._result = None

    # Getters and Setters

    def getMethodName(self) -> str:
        """ Get the Name of this Collection method """
        return str(self.__class__)

    def getReturnSize(self) -> int:
        """ Get the Number of Features that we expect to Return """
        return self._parameter

    def getOwnerPipeline(self):
        """ Get the Pipeline that owns this method """
        return self._owner

    # Public Interface

    def invoke(self,signalData,*args):
        """ Run this Collection method """
        if (Administrative.FeatureCollectionApp.getInstance().getSettings().getVerbose() > 1):
            msg = "\t\tInvoking " + self.getMethodName()
            Administrative.FeatureCollectionApp.logMessage(msg)
        #self._result = np.zeros(shape=(self.getReturnSize(),),dtype=np.float32)
        self.evaluatePreprocessCallbacks(signalData)
        return self

    def featureNames(self):
        """ Get List of Names for Each element in Result Array """
        return [self._methodName + str(i) for i in range(self.getReturnSize())]

    def registerWithPipeline(self,pipeline):
        """ Register the pipeline that owns this collection method (optional) """
        self._owner = pipeline
        return self

    def registerPreprocessCallback(self,callback):
        """ Register a preprocess Callback """
        self._preprocessCallbacks.append(callback)
        return self

    def registerPostprocessCallback(self,callback):
        """ Register a postprocess callback """
        self._postprocessCallbacks.append(callback)
        return self
    
    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        if (self._parameter <= 0):
            # Must be 1 or More
            errMsg = "Parameter must be greater than or equal to 1!"
            raise ValueError(errMsg)
        return True

    def checkForNaNsAndInfs(self):
        """ Check To See if Any Entries in the result contain NaN or Inf values """
        sumOfResult = np.sum(self._result)
        if np.isnan(sumOfResult):
            # Result contains NaN's
            msg = "\t\tMethod: {0} got result w/ NaN value(s)".format(self.getMethodName())
            Administrative.FeatureCollectionApp.getInstance().logMessage(msg)
        if np.isinf(sumOfResult):
            # Result contains NaN's
            msg = "\t\tMethod: {0} got result w/ Inf value(s)".format(self.getMethodName())
            Administrative.FeatureCollectionApp.getInstance().logMessage(msg)        
        return self

    def evaluatePreprocessCallbacks(self,signalData):
        """ Evalate the preprocess callbacks """
        for item in self._preprocessCallbacks:
            item(self,signalData)
        return self

    def evaluatePostProcessCallbacks(self,signalData):
        """ Evaluate the post process callbacks """
        for item in self._postprocessCallbacks:
            item(self,signalData)
        return self

    # Static Method

    @staticmethod
    def logExecutionTimestamp(collector,signalData):
        """ Log the Execution of this Method """
        msg = "\t\t\tRunning {0} ...".format(collector)
        Administrative.FeatureCollectionApp.getInstance().logMessage(msg)
        return None

    # Magic Methods

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + hex(id(self))

class TimeDomainEnvelopPartitions (CollectionMethod):
    """ Computes the Time-Domain-Envelope by breaking Signal into partitions """

    def __init__(self,numPartitions):
        """ Constructor for TimeDomainEnvelopPartitions Instance """
        super().__init__("TimeDomainEnvelopPartitions",numPartitions)
        self.validateParameter()

    def __del__(self):
        """ Destructor for TimeDomainEnvelopPartitions Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)  
        sizeOfPartition = signalData.Waveform.shape[0] // self._parameter
        # Iterate Through Each Parition
        startIndex = 0
        for i in range(self._parameter):    
            part = signalData.Waveform[ startIndex : startIndex + sizeOfPartition]
            self._result[i] = np.sum((part**2),dtype=np.float32)
            startIndex += sizeOfPartition
        self.checkForNaNsAndInfs()
        return self._result


    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "signalData.Waveform must not be None"
            raise ValueError(errMsg)
        if (signalData.Waveform.shape[0] < 2* self._parameter):
            errMsg = "signalData.Waveform is too small to compute TDE"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """ 
        super().validateParameter()
        if (self._parameter < 2 or self._parameter > 32):
            # Param should be greater than 1 and less than 33
            errMsg = "numParitions should be greater than 2 and less than 33"
            raise ValueError(errMsg)
        return True

class TimeDomainEnvelopFrames(CollectionMethod):
    """ Computes the TimeDomainEnvelop of Each Time-Series Analysis Frame """

    def __init__(self,startFrame=0,endFrame=256,skip=1):
        """ Constructor for TimeDomainEnvelopFrames Instance """
        numFrames = int(endFrame - startFrame) // skip
        super().__init__("TimeDomainEnvelopFrames",numFrames)
        self.validateParameter()
        self._numFrames     = numFrames
        self._start         = startFrame
        self._stop          = endFrame
        self._step          = skip

    def __del__(self):
        """ Destructor for TimeDomainEnvelopFrames Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData) 
        idx = 0
        for i in range(self._start,self._stop,self._step):
            self._result[idx] = signalData.FrameEnergyTime[i]
            idx += 1
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.FrameEnergyTime is None):
            # Make the Frame Energies
            signalData.makeFrameEnergiesTime()
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class PercentFramesAboveEnergyThreshold(CollectionMethod):
    """
    Compute the Number of Frames with energy above threshold% of Maximum energy
    """

    def __init__(self,threshold):
        """ Constructor for PercentFramesEnergyAboveThreshold Instance """
        super().__init__("PercentFramesEnergyAboveThreshold",1)
        self._thresholdFactor = threshold
        self.validateParameter()

    def __del__(self):
        """ Destructor for PercentFramesEnergyAboveThreshold Instance """
        super().__del__()

    # Getters and Setters

    def getThresholdFactor(self):
        """ Get the Threshold Factor for this instance """
        return self._thresholdFactor

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)   

        # Get Max Frame Energy + Find Threshold to beat
        maxEnergy = np.max(signalData.FrameEnergyTime)
        threshold = maxEnergy * self.getThresholdFactor()
        numFrames = 0       # number of frames above the threshold
        totFrames = signalData.FrameEnergyTime.shape[0]

        # Iterate through the Frame Energies
        for item in signalData.FrameEnergyTime:
            if (item > threshold):
                # Meets the energy criteria
                numFrames += 1

        # Get Number of Frames as a percentage
        self._result[0] = (numFrames / totFrames)
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.FrameEnergyTime is None):
            # Make the Frame Energies
            signalData.makeFrameEnergiesTime()
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class ZeroCrossingsPerTime(CollectionMethod):
    """
    Compute the total number of zero crossings normalized by signal length
    """

    def __init__(self,param=1):
        """ Constructor for ZeroCrossingsPerTime Instance """
        super().__init__("ZeroCrossingsPerTime",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for ZeroCrossingsPerTime Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)  
        
        numSamples = signalData.getNumSamples()
        signA = np.sign(signalData.Waveform[0:-2])
        signB = np.sign(signalData.Waveform[1:-1])
        outArr = np.empty(shape=signA.shape,dtype=np.float32)
        ZXR = 0

        # Iterate through Sampeles
        np.abs(signB - signA,out=outArr)
        ZXR = np.sum(outArr)
        self._result[0] = ZXR / 2
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "signalData.Waveform must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class ZeroCrossingsFramesMean(CollectionMethod):
    """
    Compute the average number of zero crossings over all analysis frames
    """

    def __init__(self,param=1):
        """ Constructor for ZeroCrossingsFramesAverage Instance """
        super().__init__("ZeroCrossingsFramesAverage",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for ZeroCrossingsFramesAverage Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)  
        self._result[0] = np.mean(signalData.FrameZeroCrossings)
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.FrameZeroCrossings is None):
            signalData.makeZeroCrossingRate()
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class ZeroCrossingsFramesVariance(CollectionMethod):
    """
    Compute the variance of zero crossings over all analysis frames
    """

    def __init__(self,param=1):
        """ Constructor for ZeroCrossingsFramesVariance Instance """
        super().__init__("ZeroCrossingsFramesVariance",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for ZeroCrossingsFramesVariance Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)
        self._result[0] = np.var(signalData.FrameZeroCrossings)
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.FrameZeroCrossings is None):
            signalData.makeZeroCrossingRate()
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class ZeroCrossingsFramesDiffMinMax(CollectionMethod):
    """
    Compute the difference of the min and max of zero crossings 
    over all analysis frames
    """

    def __init__(self,param):
        """ Constructor for ZeroCrossingsFramesDiffMinMax Instance """
        super().__init__("ZeroCrossingsFramesDiffMinMax",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for ZeroCrossingsFramesDiffMinMax Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData) 
        minVal = np.min(signalData.FrameZeroCrossings)
        maxVal = np.max(signalData.FrameZeroCrossings)
        
        self._result[0] = maxVal - minVal
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.FrameZeroCrossings is None):
            signalData.makeZeroCrossingRate()
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class TemporalCenterOfMass(CollectionMethod):
    """
    Compute the Temporal Center of Mass, weighted Quadratically
    """

    def __init__(self,kernelType="linear"):
        """ Constructor for TemporalCenterOfMass Instance """
        super().__init__("TemporalCenterOfMass",1)
        self._kernelType = kernelType.upper()
        self.validateParameter()

    def __del__(self):
        """ Destructor for TemporalCenterOfMass Instance """
        super().__del__()

    # Getters and Setters

    def getKernelType(self):
        """ Return the Type of Weighting used in the COM Calculation """
        return self._kernelType

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)   

        # Compute Total Mass + Weights
        waveformAbs = np.abs(signalData.Waveform)
        massTotal = np.sum(waveformAbs)
        weights = self.kernelFunction(signalData.getNumSamples())
        # Compute Center of Mass (By Weights)
        massCenter = np.dot(weights,waveformAbs);
        massCenter /= massTotal
        massCenter /= signalData.getNumSamples()

        # Apply Result + Return 
        self._result[0] = massCenter
        self.checkForNaNsAndInfs()
        return self._result

    def featureNames(self):
        """ Get List of Names for Each element in Result Array """
        return [self._methodName + self.kernelName() + str(i) for i in range(self.getReturnSize())]

    # Protected Interface

    def kernelFunction(self,numSamples):
        """ Set the Kernel Function based on the parameter """
        kernel = np.arange(0,numSamples,1)
        if (self._kernelType == "LINEAR"):
            pass                    # Linear Kernel
        elif (self._kernelType == "QUADRATIC"):
            kernel = kernel ** 2    # Quadratic
        elif (self._kernelType == "NATURAL_LOG"):
            kernel = np.log(kernel + EPSILON[0]) # Nat log
        else:
            pass
        return kernel

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "signalData.Samples must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class AutoCorrelationCoefficients(CollectionMethod):
    """
    Compute the First k Auto-CorrelationCoefficients
    """

    def __init__(self,numCoeffs):
        """ Constructor for AutoCorrelationCoefficients Instance """
        super().__init__("AutoCorrelationCoefficients",numCoeffs)
        self.validateParameter()

    def __del__(self):
        """ Destructor for AutoCorrelationCoefficients Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)   

        #Check is ACC's exist - make them if not
        if (signalData.AutoCorrelationCoeffs is None):
            signalData.makeAutoCorrelationCoeffs(self._parameter)

        # Copy the ACC's the the result + Return
        np.copyto(self._result,signalData.AutoCorrelationCoeffs)
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "signalData.Samples must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class AutoCorrelationCoefficientsMean(CollectionMethod):
    """
    Compute the mean of the first Auto-Correlation-Coefficients
    """

    def __init__(self,param):
        """ Constructor for AutoCorrelationCoefficientsMean Instance """
        super().__init__("AutoCorrelationCoefficientsMean",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for AutoCorrelationCoefficientsMean Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)   

        # Get the Average of the AutoCorrelation Coefficients
        self._result[0] = np.mean(signalData.AutoCorrelationCoeffs)
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AutoCorrelationCoeffs is None):
            errMsg = "signalData.AutoCorrelationCoeffs must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class AutoCorrelationCoefficientsVariance(CollectionMethod):
    """
    Compute the variance of the first Auto-Correlation-Coefficients
    """

    def __init__(self,param):
        """ Constructor for AutoCorrelationCoefficientsVariance Instance """
        super().__init__("AutoCorrelationCoefficientsVariance",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for AutoCorrelationCoefficientsVariance Instances """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData) 
        
        # Compute the Variance
        self._result[0] = np.var(signalData.AutoCorrelationCoeffs)
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AutoCorrelationCoeffs is None):
            errMsg = "signalData.AutoCorrelationCoeffs must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class AutoCorrelationCoefficientsDiffMinMax(CollectionMethod):
    """
    Compute the Different of min and max of the first Auto-Correlation-Coefficients
    """

    def __init__(self,param):
        """ Constructor for AutoCorrelationCoefficientsDiffMinMax v """
        super().__init__("AutoCorrelationCoefficientsDiffMinMax",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for AutoCorrelationCoefficientsDiffMinMax Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData) 
        # Compute Difference between Min and Max
        minVal = np.min(signalData.AutoCorrelationCoeffs)
        maxVal = np.max(signalData.AutoCorrelationCoeffs)
        self._result[0] = maxVal - minVal
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AutoCorrelationCoeffs is None):
            errMsg = "signalData.AutoCorrelationCoeffs must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class FrequencyCenterOfMass(CollectionMethod):
    """
    Compute the Frequency Center of Mass over all frames weighted linearly
    """

    def __init__(self,kernelType="linear"):
        """ Constructor for FrequencyCenterOfMass Instance """
        super().__init__(FrequencyCenterOfMass,1)
        self._kernelType = kernelType.upper()
        self.validateParameter()

    def __del__(self):
        """ Destructor for FrequencyCenterOfMassLinear Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)   

        # Compute Total Mass + Weights
        sizeOfFrame = signalData.AnalysisFramesFreq.shape[1]
        massTotal = np.sum(signalData.AnalysisFramesFreq,axis=-1) + EPSILON
        weights = self.kernelFunction(sizeOfFrame)
       
        # Compute Center of Mass (by Weights)
        massCenter = np.matmul(signalData.AnalysisFramesFreq,weights)
        massCenter /= massTotal
        massCenter /= sizeOfFrame

        # Add the Average of all frames, and put into result
        self._result[0] = np.mean(massCenter)
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def kernelFunction(self,numSamples):
        """ Set the Kernel Function based on the parameter """
        kernel = np.arange(0,numSamples,1)
        if (self._kernelType == "LINEAR"):
            pass                    # Linear Kernel
        elif (self._kernelType == "QUADRATIC"):
            kernel = kernel ** 2    # Quadratic
        elif (self._kernelType == "NATURAL_LOG"):
            kernel = np.log(kernel + EPSILON[0]) # Nat log
        else:
            pass
        return kernel

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AnalysisFramesFreq is None):
            errMsg = "signalData.AnalysisFramesFreq must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class MelFilterBankEnergies(CollectionMethod):
    """
    Compute K Mel Frequency Cepstrum Coefficients
    """

    def __init__(self,numCoeffs):
        """ Constructor for MelFrequencyCempstrumCoeffs Instance """
        super().__init__("MelFilterBankEnergies",numCoeffs)
        self.validateParameter()

    def __del__(self):
        """ Destructor for MelFrequencyCempstrumCoeffs Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)   

        # Check if We have MFCC's - Create if we don't
        if (signalData.MelFilterBankEnergies is None):
            signalData.makeMelFilterBankEnergies(
                self._owner.getAnalysisFrameParams(),self._parameter)
        avgMFBEs = np.mean(signalData.MelFilterBankEnergies,axis=0)

        # Copy to result + return
        np.copyto(self._result,avgMFBEs)
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AnalysisFramesFreq is None):
            errMsg = "signalData.AnalysisFramesFreq must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

    # Static Interface

    @staticmethod
    def melsToHertz(freqMels):
        """ Cast Mels Samples to Hertz Samples """
        return 700 * ( 10** (freqMels / 2595) -1 )

    @staticmethod
    def hertzToMels(freqHz):
        """ Cast Hertz Samples to Mels Samples """
        return 2595 * np.log10(1 + freqHz / 700)

    @staticmethod
    def melFilters(frameParams,numFilters,sampleRate=44100):
        """ Build the Mel-Filter Bank Arrays """
        freqBoundsHz = frameParams.getFreqBoundHz()
        freqBoundsMels = MelFilterBankEnergies.hertzToMels(freqBoundsHz)
        numSamplesTime = frameParams.getTotalTimeFrameSize()       

        freqAxisMels = np.linspace(freqBoundsMels[0],freqBoundsMels[1],numFilters+2)
        freqAxisHz = MelFilterBankEnergies.melsToHertz(freqAxisMels)
        bins = np.floor((numSamplesTime+1)*freqAxisHz/sampleRate)
        filterBanks = np.zeros(shape=(numFilters,numSamplesTime),dtype=np.float32)

        # Iterate through filters
        for i in range (1,numFilters + 1,1):
            freqLeft = int(bins[i-1])
            freqCenter = int(bins[i])
            freqRight = int(bins[i+1])

            for j in range(freqLeft,freqCenter):
                filterBanks[i-1,j] = (j - freqLeft) / (freqCenter - freqLeft)
            for j in range(freqCenter,freqRight):
                filterBanks[i-1,j] = (freqRight - j) / (freqRight - freqCenter)

        # Crop to Subset of Frequency Space
        numSamplesFreq = frameParams.getFreqFramesShape()[1]
        filterBanks = filterBanks[:,:numSamplesFreq]
        return filterBanks


class MelFilterBankEnergiesMean(CollectionMethod):
    """
    Compute Average of Mel Frequency Cepstrum Coefficients
    """

    def __init__(self,numCoeffs):
        """ Constructor for MelFrequencyCempstrumCoeffsMean Instance """
        super().__init__("MelFilterBankEnergiesMean",numCoeffs)
        self.validateParameter()

    def __del__(self):
        """ Destructor for MelFrequencyCempstrumCoeffsMean Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)   
        # Compute Mean of MFCC's
        avgMFBEs = np.mean(signalData.MelFilterBankEnergies,axis=0)
        self._result[0] = np.mean(avgMFBEs)
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.MelFilterBankEnergies is None):
            errMsg = "signalData.MelFilterBankEnergies must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class MelFilterBankEnergiesVariance(CollectionMethod):
    """
    Compute variance of Mel Frequency Cepstrum Coefficients
    """

    def __init__(self,param):
        """ Constructor for MelFrequencyCempstrumCoeffsVariance Instance """
        super().__init__("MelFilterBankEnergiesVariance",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for MelFrequencyCempstrumCoeffsVariance Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)   
        # Compute Variance of MFCC's
        avgMFBEs = np.mean(signalData.MelFilterBankEnergies,axis=0)
        self._result[0] = np.var(avgMFBEs)
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.MelFilterBankEnergies is None):
            errMsg = "signalData.MelFilterBankEnergies must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class MelFilterBankEnergiesDiffMinMax(CollectionMethod):
    """
    Compute difference of min and max Mel Frequency Cepstrum Coefficients
    """

    def __init__(self,param):
        """ Constructor for MelFrequencyCempstrumCoeffsDiffMinMax Instance """
        super().__init__("MelFilterBankEnergiesDiffMinMax",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for MelFrequencyCempstrumCoeffsDiffMinMax Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)   
        # Compute Diff of min and max of MFCC's
        avgMFBEs = np.mean(signalData.MelFilterBankEnergies,axis=0)
        minVal = np.min(avgMFBEs)
        maxVal = np.max(avgMFBEs)

        self._result[0] = maxVal - minVal
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.MelFilterBankEnergies is None):
            errMsg = "signalData.MelFilterBankEnergies must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class MelFrequencyCepstrumCoefficients(CollectionMethod):
    """
    Compute the Mel-Frequency Cepstrum Coeffs
    """

    def __init__(self,param):
        """ Constructor for MelFrequencyCempstrumCoeffsDiffMinMax Instance """
        super().__init__("MelFrequencyCepstrumCoefficients",param)
        self.validateParameter()

    def __del__(self):
        """ Destructor for MelFrequencyCempstrumCoeffsDiffMinMax Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData)   
        # Compute Diff of min and max of MFCC's
        logFilterBanks = np.log(signalData.MelFilterBankEnergies + EPSILON[0])
        MFCCs = fftpack.idct(logFilterBanks,type=2,axis=-1)

        avgMFCCs = np.mean(MFCCs,axis=0)
        np.copyto(self._result,avgMFCCs)
        self.checkForNaNsAndInfs()
        return self._result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.MelFilterBankEnergies is None):
            errMsg = "signalData.MelFilterBankEnergies must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class Spectrogram(CollectionMethod):
    """ Compute the Spectrogram Representation of an Waveform """

    def __init__(self,frameParams):
        """ Constructor for Spectrogram instance """
        returnSize = frameParams.getTotalFreqFrameSize() * frameParams.getMaxNumFrames()
        super().__init__("Spectrogram",returnSize)
        self._framesParams = frameParams

    def __del__(self):
        """ Destructor for Spectrogram instance """
        super().__del__()

    # Public Interface 

    def invoke(self,signalData,*args):
        """ Invoke this collection method """
        self.validateInputSignal(signalData)
        super().invoke(signalData);
        # Build spectrogram
        if (signalData.AnalysisFramesFreq is None):
            # Spect does not exist
            signalData.makeAnalysisFramesFreq(self._framesParams)


        return self._result


    # Protected Interface

    # 