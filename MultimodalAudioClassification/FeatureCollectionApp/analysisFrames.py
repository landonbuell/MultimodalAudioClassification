"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       analysisFrames.py
    Classes:    AnalysisFrameParameters,
                AnalysisFrames,
                TimeSeriesAnalysisFrames,
                FreqSeriesAnalysisFrames

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import numpy as np
import scipy.fftpack as fftpack

        #### CLASS DEFINITIONS ####

class AnalysisFrameParameters:
    """ Stores parameters for analysis frames """
 
    def __init__(self,
                 samplesPerFrame=1024,
                 sampleOverlap=768,
                 headPad=1024,
                 tailPad=2048,
                 maxNumFrames=512,
                 freqLowBoundHz=0.0,
                 freqHighBoundHz=12000.0):
        """ Constructor """
        self.samplesPerFrame    = samplesPerFrame
        self.sampleOverlap      = sampleOverlap
        self.headPad            = headPad
        self.tailPad            = tailPad
        self.maxNumFrames       = maxNumFrames

        self.freqHighBoundHz = freqHighBoundHz
        self.freqLowBoundHz = freqLowBoundHz

    def __del__(self):
        """ Destructor """
        pass

    @staticmethod
    def defaultFrameParams():
        """ Return the default analysis frame parameters """
        return AnalysisFrameParameters()

    # Accessors

    @property
    def timeFrameSize(self) -> int:
        """ Get the total size of each frame """
        return self.samplesPerFrame

    @property
    def freqFrameSizeUnCropped(self) -> int:
        """ Get the number of samples an uncropped frequency spectrum """
        return self.headPad + self.samplesPerFrame + self.tailPad

    @property
    def freqFrameSizeCropped(self) -> int:
        """ Get the total size of each frame """
        return len(self.getMaskedFrequencyAxisHz())

    @property
    def freqHighBoundMels(self) -> float:
        """ Return the frequency high bound in Mels """
        return AnalysisFrameParameters.hzToMels(self.freqHighBoundHz)

    @property
    def freqLowBoundMels(self) -> float:
        """ Return the frequency low bound in Mels """
        return AnalysisFrameParameters.hzToMels(self.freqLowBoundHz)
   
    def getFreqFramesShape(self) -> tuple:
        """ Return the SHAPE of the frequency frames """
        return (self.maxNumFrames,self.freqFrameSize,)

    def getTimeFramesShape(self) -> tuple:
        """ Return the SHAPE of the time frames """
        return (self.maxNumFrames,self.timeFrameSize,)

    def getFreqFramesNumFeatures(self) -> int:
        """ Return the total number of data points in the frequency frames """
        return (self.maxNumFrames * self.freqFrameSize)

    def getUnmaskedFrequencyAxisHz(self) -> np.ndarray:
        """ Return an uncropped frequency axis """
        sampleSpacing = 1.0 / AnalysisFrameParameters.sampleRate()
        freqAxis = np.fft.fftfreq(n=self.timeFrameSize,
                                  d=sampleSpacing)
        return freqAxis

    def getMaskedFrequencyAxisHz(self) -> np.ndarray:
        """ Return a frequency axis cropped by provided bounds """
        sampleSpacing = 1.0 / AnalysisFrameParameters.sampleRate()
        freqAxis = np.fft.fftfreq(n=self.timeFrameSize,
                                  d=sampleSpacing)
        freqMask = (freqAxis >= self.freqLowBoundHz) and (freqAxis <= self.freqHighBoundHz)
        freqAxis = freqAxis[freqMask]   # apply mask
        return freqAxis

    # Static Interface

    @staticmethod
    def melToHz(freqMels: np.ndarray) -> np.ndarray:
        """ Cast Mel Frequency to Hz """
        return 700.0 *  (np.power(10,(freqMels/2595)) - 1)

    @staticmethod
    def hzToMels(freqHz: np.ndarray) -> np.ndarray:
        """ Cast Hz Frequency to Mels """
        return 2595 * np.log10(1 + (freqHz/700))

    @staticmethod
    def sampleRate() -> float:
        """ Return sample Rate. TEMP HARD-CODED """
        return 44100.0

    # Magic Methods

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))

    def __eq__(self,other) -> bool:
        """ Implement Equality Operator """
        eq = (  (self.samplesPerFrame == other.samplesPerFrame) and
                (self.sampleOverlap == other.sampleOverlap) and
                (self.headPad == other.headPad) and
                (self.tailPad == other.tailPad) and 
                (self.maxNumFrames == other.maxNumFrames) and 
                (self.freqHighBoundHz == other.freqHighBoundHz) and 
                (self.freqLowBoundHz == other.freqLowBoundHz))
        return eq

    def __neq__(self,other) -> bool:
        """ Implement inequality operator """
        return not self.__eq__(other)

class __AbstractAnalysisFrames:
    """ Abstract Base Class for All Analysis Frame Types """

    def __init__(self,
                 signalData,
                 frameParams: AnalysisFrameParameters,
                 numFrames: int,
                 frameSize: int,
                 dataType: type):
        """ Constructor """
        self._params = frameParams
        self._data   = np.zeros(shape=(numFrames,frameSize),dtype=dataType)
        self._framesInUse = 0
        self.populate(signalData)

    def __del__(self):
        """ Destructor """
        self._params = None
        self._data  = None

    # Accessors

    def getParams(self) -> AnalysisFrameParameters:
        """ Return the parameters that constructed these analysis frames """
        return self._params

    def getMaxNumFrames(self) -> int:
        """ Get the MAX number of analysis frames """
        return self._data.shape[0]

    def getNumFramesInUse(self) -> int:
        """ Get the number of analysis frames in use """
        return self._framesInUse

    def getFrameSize(self) -> int:
        """ Get the size of each frame """
        return self._data.shape[1]

    def rawFrames(self,onlyInUse=False) -> np.ndarray:
        """ Return the raw underlying analysis frames """
        if (onlyInUse == True):
            return self._data[0:self._framesInUse]
        return self._data

    # Public Interface
    
    def populate(self,
                 signalData) -> None:
        """ Populate the analysis frames """
        success = self._validateSignal(signalData)
        if (success == True):
            self._populateFrames(signalData)
        return None

    def clear(self) -> None:
        """ Zero all frame values """
        for ii in range(self._data.shape[0]):
            for jj in range(self._data.shape[1]):
                self._data[ii,jj] = 0.0
        self._framesInUse = 0
        return None

    # Protected Interface

    def _validateSignal(self,
                        signalData) -> bool:
        """ VIRTUAL: Validate that the input signal has info to work with """
        if (signalData.getNumSamples() == 0):
            errMsg = "provided signal: {0} has {1} samples.".format(
                repr(signalData),signalData.getNumSamples())
            raise RuntimeError(errMsg)
        return True

    def _populateFrames(self,
                        signalData) -> None:
        """ VIRTUAL: Populate the analysis frames """     
        return None
    
    # Magic Methods

    def __getitem__(self,
                    key: int) -> np.ndarray:
        """ Return the frame at the provided value """
        return self._data[key]

    def __setitem__(self,
                    key: int,
                    val: np.ndarray) -> None:
        """ Set the provided value at the provided frame key """
        self._data[key] = val
        return None

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))


class TimeSeriesAnalysisFrames(__AbstractAnalysisFrames):
    """ Stores Short-time-series analysis frames """

    __DATA_TYPE = np.float32

    def __init__(self,
                 signalData,
                 frameParams: AnalysisFrameParameters):
        """ Constructor """
        super().__init__(   signalData,
                            frameParams,
                            frameParams.maxNumFrames,
                            frameParams.timeFrameSize,
                            TimeSeriesAnalysisFrames.__DATA_TYPE)

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Protected Interface

    def _validateSignal(self,
                        signalData) -> bool:
        """ VIRTUAL: Validate that the input signal has info to work with """
        valid = super()._validateSignal(signalData)
        return valid

    def _populateFrames(self,
                        signalData) -> None:
        """ OVERRIDE: Populate the analysis frames """
        stepSize    = self._params.samplesPerFrame - self._params.sampleOverlap
        frameStart  = 0
        frameEnd    = 0

        for ii in range(self.getMaxNumFrames()):
            frameEnd = frameStart + self._params.samplesPerFrame
            if (frameEnd > len(signalData)):
                if (frameStart > len(signalData)):
                    break
                frameEnd = len(signalData) - 1
                frameSlice = signalData[frameStart:frameEnd]
                self._data[ii,0:frameSlice.size] = frameSlice
            else:
                # Store the items in the frame
                self._data[ii] = signalData[frameStart:frameEnd]
            # increment the front + end
            frameStart  += stepSize
            frameEnd    = frameStart + self._params.samplesPerFrame
            self._framesInUse += 1
        return None

class FreqSeriesAnalysisFrames(__AbstractAnalysisFrames):
    """ Stores Short-time-frequency-series analysis frames """

    __DATA_TYPE = np.complex64

    def __init__(self,
                 signalData,
                 frameParams: AnalysisFrameParameters,
                 multiThread=False):
        """ Constructor """
        super().__init__(signalData,
                         frameParams,
                         signalData.cachedData.analysisFramesTime.getNumFramesInUse(),
                         frameParams.freqFrameSize,
                         FreqSeriesAnalysisFrames.__DATA_TYPE)
        self._freqAxis      = self._params.getMaskedFrequencyAxisHz()
        self._multiThread   = multiThread

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def useMultipleThreads(self) -> bool:
        """ Return T/F if we should use multiple threads """
        return self._multiThread

    # Protected Interface

    def _validateSignal(self,
                        signalData) -> bool:
        """ VIRTUAL: Validate that the input signal has info to work with """
        valid = super()._validateSignal(signalData)
        if (valid == True):
            valid = (signalData.cachedData.analysisFramesTime is not None)
        return valid

    def _populate(self,
                  signalData) -> None:
        """ OVERRIDE: Populate the analysis frames """
        numTimeFrames = signalData.cachedData.analysisFramesTime.getNumFramesInUse()
        rawTimeFrames = signalData.cachedData.analysisFramesTime.rawFrames()
        maskFreqAxis = self._params.getMaskedFrequencyAxisHz()
        fftData = np.fft.fft( rawTimeFrames )    
        self._data = fftData[:,maskFreqAxis] # apply mask
        return None