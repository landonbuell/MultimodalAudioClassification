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

import signalData

        #### CLASS DEFINITIONS ####

class AnalysisFrameParameters:
    """ Stores parameters for analysis frames """
 
    def __init__(self,
                 samplesPerFrame=1024,
                 sampleOverlap=768,
                 headPad=1024,
                 tailPad=2048,
                 maxNumFrames=256,
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

    # Accessors

    @property
    def timeFrameSize(self) -> int:
        """ Get the total size of each frame """
        return self.headPad + self.samplesPerFrame + self.tailPad

    @property
    def freqFrameSize(self) -> int:
        """ Get the total size of each frame """
        # TODO: Implement this!
        return self.headPad + self.samplesPerFrame + self.tailPad

    @property
    def freqHighBoundMels(self) -> float:
        """ Return the frequency high bound in Mels """
        return AnalysisFrameParameters.hzToMels(self.freqHighBoundHz)

    @property
    def freqLowBoundMels(self) -> float:
        """ Return the frequency low bound in Mels """
        return AnalysisFrameParameters.hzToMels(self.freqLowBoundHz)

    def getUnmaskedFrequencyAxisHz(self) -> np.ndarray:
        """ Return an uncropped frequency axis """
        sampleSpacing = 1.0 / AnalysisFrameParameters.sampleRate
        freqAxis = np.fft.fftfreq(n=self.timeFrameSize,
                                  d=sampleSpacing)
        return freqAxis

    def getMaskedFrequencyAxisHz(self) -> np.ndarray:
        """ Return a frequency axis cropped by provided bounds """
        sampleSpacing = 1.0 / AnalysisFrameParameters.sampleRate
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
        """ Return sample Rate """
        return 44100.0

    # Magic Methods

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))


class __AbstractAnalysisFrames:
    """ Abstract Base Class for All Analysis Frame Types """

    def __init__(self,
                 signal : signalData.SignalData,
                 frameParams: AnalysisFrameParameters,
                 frameSize: int,
                 dataType: type):
        """ Constructor """
        self._params = frameParams
        self._data   = np.zeros(shape=(frameParams.maxNumFrames,frameSize),dtype=dataType)
        
        self._validateSignal(signal)
        self._populateFrames(signal)

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getParams(self) -> AnalysisFrameParameters:
        """ Return the parameters that constructed these analysis frames """
        return self._params

    def getNumFrames(self) -> int:
        """ Get the number of analysis frames """
        return self._data.shape[0]

    def getFrameSize(self) -> int:
        """ Get the size of each frame """
        return self._data.shape[1]

    def rawFrames(self) -> np.ndarray:
        """ Return the raw underlying analysis frames """
        return self._data

    # Protected Interface

    def _validateSignal(self,
                        signal: signalData.SignalData) -> None:
        """ VIRTUAL: Validate that the input signal has info to work with """
        if (signal.getNumSamples() == 0):
            errMsg = "provided signal: {0} has {1} samples.".format(
                repr(signal),signal.getNumSamples())
            raise RuntimeError(errMsg)
        return None

    def _populateFrames(self,
                        signal: signalData.SignalData) -> None:
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
                 signal : signalData.SignalData,
                 frameParams: AnalysisFrameParameters):
        """ Constructor """
        super().__init__(   signal,
                            frameParams,
                            frameParams.timeFrameSize,
                            TimeSeriesAnalysisFrames.__DATA_TYPE)

    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    def populate(self,
                 signal: signalData.SignalData ) -> None:
        """ OVERRIDE: Populate the analysis frames """
        stepSize = self._params.samplesPerFrame - self._params.sampleOverlap
        frameStart = 0
        frameEnd = self._params.samplesPerFrame

        for ii in range(self.getNumFrames()):
            if (frameEnd > len(signal)):
                if (frameStart > len(signal)):
                    break
                frameEnd = len(signal)
            # Grab the items
            frameData = signal[frameStart:frameEnd];
            self._data[ii,self._params.headPad:self._params.headPad + self._params.samplesPerFrame] = frameData
            # increment the front + end
            frameStart += stepSize
            frameEnd = frameStart + self._params.samplesPerFrame
        return None


class FreqSeriesAnalysisFrames(__AbstractAnalysisFrames):
    """ Stores Short-time-frequency-series analysis frames """

    __DATA_TYPE = np.complex64

    def __init__(self,
                 signal : signalData.SignalData,
                 frameParams: AnalysisFrameParameters):
        """ Constructor """
        super().__init__(frameParams,
                         frameParams.freqFrameSize,
                         FreqSeriesAnalysisFrames.__DATA_TYPE)
        self._freqAxis = self._params.getMaskedFrequencyAxisHz()

    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    def _validateSignal(self,
                        signal: signalData.SignalData) -> None:
        """ VIRTUAL: Validate that the input signal has info to work with """
        if (signal.getCachedData().analysisFramesTime == None):
            # No time-frames yet ... make them!
            signal.populateTimeSeriesAnalysisFrames()
        return None

    def _populate(self,
                  signal: signalData.SignalData) -> None:
        """ OVERRIDE: Populate the analysis frames """
        rawTimeFrames = signal.getCachedData().analysisFramesTime.getRawFrames()
        self._data = np.fft.fft( rawTimeFrames )
        # TODO: Apply Freq Mask
        # TODO: Finish this!
        return None
