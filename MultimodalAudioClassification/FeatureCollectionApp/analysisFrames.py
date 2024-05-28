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
import matplotlib.pyplot as plt
import threading

        #### FUNCTION DEFINITIONS ####

def debugPlot(yData,title):
    """ Show Time-Series Signal for quick debugging"""
    plt.figure(figsize=(12,8))
    plt.title(title,fontsize=32,fontweight='bold')
    plt.xlabel("SampleIndex",fontsize=24,fontweight='bold')
    plt.ylabel("Amplitude",fontsize=24,fontweight='bold')

    plt.plot(yData,label="Data")
    plt.hlines(y=[0],xmin=0,xmax=yData.size,color='black')

    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()
    return None

def debugPlotXy(xData,yData,title):
    """ Show Time-Series Signal for quick debugging"""
    plt.figure(figsize=(12,8))
    plt.title(title,fontsize=32,fontweight='bold')
    plt.xlabel("SampleIndex",fontsize=24,fontweight='bold')
    plt.ylabel("Amplitude",fontsize=24,fontweight='bold')

    plt.plot(xData,yData,label="Data")
    plt.hlines(y=[0],xmin=np.min(xData),xmax=np.max(xData),color='black')

    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()
    return None

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
                 freqHighBoundHz=16010.0,
                 window=np.hanning):
        """ Constructor """
        self.samplesPerFrame    = samplesPerFrame
        self.sampleOverlap      = sampleOverlap
        self.headPad            = headPad
        self.tailPad            = tailPad
        self.maxNumFrames       = maxNumFrames

        self.freqHighBoundHz    = freqHighBoundHz
        self.freqLowBoundHz     = freqLowBoundHz

        self.window         = window(samplesPerFrame)
        self._melFilters    = dict() # int -> np.ndarray

    def __del__(self):
        """ Destructor """
        pass

    @staticmethod
    def defaultFrameParams():
        """ Return the default analysis frame parameters """
        return AnalysisFrameParameters()

    # Accessors

    @property
    def freqHighBoundMels(self) -> float:
        """ Return the frequency high bound in Mels """
        return AnalysisFrameParameters.hzToMel(self.freqHighBoundHz)

    @property
    def freqLowBoundMels(self) -> float:
        """ Return the frequency low bound in Mels """
        return AnalysisFrameParameters.hzToMel(self.freqLowBoundHz)

    def getTimeFrameSize(self) -> int:
        """ Get the total size of each frame """
        return self.samplesPerFrame

    def getFreqFrameSizeUnmasked(self) -> int:
        """ Get the number of samples an uncropped frequency spectrum """
        return self.headPad + self.samplesPerFrame + self.tailPad

    def getFreqFrameSizeMasked(self) -> int:
        """ Get the total size of each frame """
        return self.getFreqAxisMask().size

    def getFreqFramesShape(self) -> tuple:
        """ Return the SHAPE of the frequency frames """
        return (self.maxNumFrames,self.getFreqFrameSizeMasked(),)

    def getTimeFramesShape(self) -> tuple:
        """ Return the SHAPE of the time frames """
        return (self.maxNumFrames,self.getTimeFrameSize(),)

    def getFreqFramesNumFeatures(self,separateRealImag=False) -> int:
        """ Return the total number of data points in the frequency frames """
        result = (self.maxNumFrames * self.getFreqFrameSizeMasked())
        if (separateRealImag == True):
            result *= 2
        return result

    def getFreqFrameShape(self,separateRealImag=False) -> tuple:
        """ Return the shape of the frequency frames. Option to separate real/imag """
        if (separateRealImag  == True):
            return (2,self.maxNumFrames,self.getFreqFrameSizeMasked(),)
        return (1,self.maxNumFrames,self.getFreqFrameSizeMasked(),)


    def getFreqAxisMask(self) -> np.ndarray:
        """ Return the mask for the frequency axis """
        sampleSpacing = 1.0 / AnalysisFrameParameters.sampleRate()
        freqAxis = np.fft.fftfreq(n=self.getFreqFrameSizeUnmasked(),
                                  d=sampleSpacing)
        mask = np.where((freqAxis >= self.freqLowBoundHz) & (freqAxis < self.freqHighBoundHz))[0]
        return mask

    def getFreqAxisUnmasked(self) -> np.ndarray:
        """ Return an uncropped frequency axis """
        sampleSpacing = 1.0 / AnalysisFrameParameters.sampleRate()
        freqAxis = np.fft.fftfreq(n=self.getFreqFrameSizeUnmasked(),
                                  d=sampleSpacing)
        return freqAxis

    def getFreqAxisMasked(self) -> np.ndarray:
        """ Return a cropped frequency axis """
        sampleSpacing = 1.0 / AnalysisFrameParameters.sampleRate()
        freqAxis = np.fft.fftfreq(n=self.getFreqFrameSizeUnmasked(),
                                  d=sampleSpacing)
        mask = np.where((freqAxis >= self.freqLowBoundHz) & (freqAxis < self.freqHighBoundHz))[0]
        return freqAxis[mask]

    def getMelFilters(self,numFilters: int) -> np.ndarray:
        """ Return the Mel Filter banks """
        if (self._melFilters.get(numFilters,None) is None):
            self._melFilters[numFilters] = self.__createMelFilters(numFilters)
        return self._melFilters[numFilters] # hash-map is O(1) lookup

    # Private Interface

    def __createMelFilters(self,numFilters: int) -> None:
        """ Create + Return mel filter banks """
        lowerFreqMels = self.freqLowBoundMels
        upperFreqMels = self.freqHighBoundMels
        melPoints = np.linspace(lowerFreqMels,upperFreqMels,numFilters + 2)
        hzPoints = AnalysisFrameParameters.melToHz(melPoints)

        frameSize = self.getFreqFrameSizeUnmasked()
        maskFreqAxisHz = self.getFreqAxisMask()

        bins = np.floor((frameSize + 1) * hzPoints / AnalysisFrameParameters.sampleRate() )
        filterBanks = np.zeros(shape=(numFilters,frameSize),dtype=np.float32)

        for ii in range(1, numFilters + 1, 1): 
            # Each filter
            freqLeft    = int(bins[ii - 1])
            freqRight   = int(bins[ii + 1])
            freqCenter  = int(bins[ii])

            for jj in range(freqLeft,freqCenter):
                filterBanks[ii-1,jj] = (jj - bins[ii-1]) / (bins[ii] - bins[ii - 1])
            for jj in range(freqCenter,freqRight):
                filterBanks[ii-1,jj] = (bins[ii+1] - jj) / (bins[ii + 1] - bins[ii])
        
        # Apply mask to frequency Axis
        filterBanks = filterBanks[:,maskFreqAxisHz]
        return filterBanks

    @staticmethod
    def plotFilters(filterMatrix: np.ndarray, freqAxis: np.ndarray) -> None:
        """ Plot all filters """
        plt.figure(figsize=(16,12))
        plt.title("Mel Filters",size=24,weight='bold')
        plt.xlabel("Frequency",size=20,weight='bold')
        plt.ylabel("Filter Strength",size=20,weight='bold')

        # Plot the Stuff
        numFilters = filterMatrix.shape[0]
        for ii in range(numFilters):
            plt.plot(freqAxis,filterMatrix[ii],label="Filter{0}".format(ii))

        # House Keeping
        plt.grid()
        plt.legend()
        plt.show()
        return None

    # Static Interface

    @staticmethod
    def melToHz(freqMels: np.ndarray) -> np.ndarray:
        """ Cast Mel Frequency to Hz """
        return 700.0 *  (np.power(10,(freqMels/2595)) - 1)

    @staticmethod
    def hzToMel(freqHz: np.ndarray) -> np.ndarray:
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
                (self.freqLowBoundHz == other.freqLowBoundHz) )
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
        # Call "self.populate()" the child constructor

    def __del__(self):
        """ Destructor """
        self._params    = None
        self._data      = None

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
            errMsg = "Provided signal: {0} has {1} samples.".format(
                repr(signalData),signalData.getNumSamples())
            raise RuntimeWarning(errMsg)
            return False
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
                            frameParams.getTimeFrameSize(),
                            TimeSeriesAnalysisFrames.__DATA_TYPE)
        self.populate(signalData)

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
                         frameParams.getFreqFrameSizeMasked(),
                         FreqSeriesAnalysisFrames.__DATA_TYPE)
        self._framesInUse   = signalData.cachedData.analysisFramesTime.getNumFramesInUse()
        self._freqMask      = self._params.getFreqAxisMask()
        self._multiThread   = multiThread
        self.populate(signalData)
        
    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def useMultipleThreads(self) -> bool:
        """ Return T/F if we should use multiple threads """
        return self._multiThread

    def getMaskedFrequencyAxisHz(self) -> np.ndarray:
        """ Return Masked frequency axis """
        return self._params.getFreqAxisMasked()

    # Public Interface

    def plot(self,title: str) -> None:
        """ Show a figure of the time-series analysis frames """
        plt.figure(figsize=(16,12))
        plt.title(title,size=32,weight='bold')
        plt.xlabel('Frequency',size=24,weight='bold')
        plt.ylabel('Time',size=24,weight='bold')

        f = self.getMaskedFrequencyAxisHz()
        t = np.arange(self.getNumFramesInUse(),0,-1)
        X = np.abs(self._data)
        plt.pcolormesh(f,t,X,cmap=plt.cm.plasma)

        plt.grid()
        plt.tight_layout()
        plt.show()

        return None

    # Protected Interface

    def _validateSignal(self,
                        signalData) -> bool:
        """ VIRTUAL: Validate that the input signal has info to work with """
        if (super()._validateSignal(signalData) == False):
            return False
        if (signalData.cachedData.analysisFramesTime is None):
            errMsg = "Provided signal does not have time-series analysis frames"
            raise RuntimeWarning(errMsg)
            return False
        if (signalData.cachedData.analysisFramesTime.getParams() != self._params):
            errMsg = "Provided signal's time-series analysis frames parmas do NOT match this one's"
            raise RuntimeWarning(errMsg)
            return False
        return True

    def _populateFrames(self,
                  signalData) -> None:
        """ OVERRIDE: Populate the analysis frames """
        if (self.useMultipleThreads == True):
            self.__populateWithMultipleThreads(signalData)
        else:
            self.__populateWithSingleThread(signalData)
        return None

    def __populateWithMultipleThreads(self,
                                     signalData) -> None:
        """ Populate frequency series analysis frames in multiple threads """
        # TODO: Implement this later
        return None

    def __populateWithSingleThread(self,
                                   signalData) -> None:
        """ Populate frequency Series analysis frames in a single thread """
        freqAxis = self._params.getFreqAxisMasked()
        for ii in range(self._framesInUse):
            self._data[ii] = self.__transform(signalData.cachedData.analysisFramesTime[ii])
            #debugPlotXy(freqAxis,self._data[ii],"Freq Frame" + str(ii))
        return None

    def __transform(self,
                    rawTimeFrame: np.ndarray) -> np.ndarray:
        """ Perform transform on signal """
        if (rawTimeFrame.size != self._params.samplesPerFrame):
            msg = "Expected {0} samples in frame but got {1}".format(
                self._params.samplesPerFrame,rawTimeFrame.size)
            raise RuntimeError(msg)
        paddedFrame = np.zeros(shape=(self._params.getFreqFrameSizeUnmasked(),),dtype=np.float32)
        rawTimeFrame *= self._params.window
        paddedFrame[self._params.headPad:self._params.headPad + rawTimeFrame.size] = rawTimeFrame
        fftData = np.fft.fft(a=paddedFrame)
        fftData = np.abs(fftData)**2 # Compute "abs" of data and element-wise square
        return fftData[self._freqMask]

class MelFilterBankEnergies:
    """ Stored Mel Filter Bank Energies """

    def __init__(self,
                 signal,
                 frameParams,
                 numFilters: int):
        """ Constructor """
        self._params        = frameParams
        self._filterMatrix  = self._params.getMelFilters(numFilters)
        self._data          = None

        self.__validateSignal(signal)

        numFrames = signal.cachedData.analysisFramesFreq.getNumFramesInUse()
        self._data = np.zeros(shape=(numFrames,numFilters))

        self.__applyMelFilters(signal)
        

    def __del__(self):
        """ Destructor """
        self._params    = None
        self._data      = None

    # Accessors

    def getParams(self) -> AnalysisFrameParameters:
        """ Return the parameters structure used to create this instance """
        return self._params

    @property
    def numFilters(self) -> int:
        """ Return the number of Mel Filter Banks """
        return self._filterMatrix.shape[0]

    @property
    def filterSize(self) -> int:
        """ Return the size of each Mel Filter """
        return self._filterMatrix.shape[1]

    def getEnergies(self) -> np.ndarray:
        """ Return the raw MFBE array """
        return self._data

    def getMeans(self) -> np.ndarray:
        """ Return mean energy of each filter """
        return np.mean(self._data,axis=0)

    def getVariances(self) -> np.ndarray:
        """ Return variance of energy in each filter """
        return np.var(self._data,axis=0)

    def getMedian(self) -> np.ndarray:
        """ Return the median energy of each filter bank """
        return np.median(self._data,axis=0)

    def getMin(self) -> np.ndarray:
        """ Return the minimum energy of each filter bank """
        return np.min(self._data,axis=0)

    def getMax(self) -> np.ndarray:
        """ Return the maximim energy of each filter bank """
        return np.max(self._data,axis=0)

    # Private Interface

    def __validateSignal(self,
                        signalData) -> bool:
        """ Validate that the input signal has info to work with """
        if (signalData.cachedData.analysisFramesTime is None):
            errMsg = "Provided signal does not have time-series analysis frames"
            raise RuntimeWarning(errMsg)
            return False
        if (signalData.cachedData.analysisFramesTime.getParams() != self._params):
            errMsg = "Provided signal's analysis frames parmas do NOT match this one's"
            raise RuntimeWarning(errMsg)
            return False
        return True

    def __applyMelFilters(self,signal) -> np.ndarray:
        """ Apply mel Filters to freq-series frames """
        # Each ROW is a filter      
        frameSize   = signal.cachedData.analysisFramesFreq.getFrameSize()
        numFrames =  signal.cachedData.analysisFramesFreq.getNumFramesInUse()
        if (self.filterSize != frameSize):
            msg = "ERROR: provided mel filters have size={1} and analysis frames have size={1}".format(
                self.filterSize,frameSize)
            raise RuntimeError(msg)
        # Pre-allocate the OUTPUT array
        freqFrames = np.abs(signal.cachedData.analysisFramesFreq.rawFrames(onlyInUse=True))**2
        melFiltersTransposed = self._filterMatrix.transpose()
        np.matmul(freqFrames,melFiltersTransposed,out=self._data)
        return None

    # Magic Methods

    def __getitem__(self,index) -> object:
        """ Return item at index """
        return self._data[index]

class MelFrequencyCepstralCoefficients:
    """ Stores the Mel Frequency Cepstral Coefficients """

    def __init__(self,
                 melFilterBankEnergies: MelFilterBankEnergies):
        """ Constructor """
        self._data = np.copy(melFilterBankEnergies.getEnergies())
        self.__createCepstralCoeffs()

    def __del__(self):
        """ Destructor """
        self._data = None

    # Accessors

    @property
    def numCoeffs(self) -> int:
        """ Return the number of Mel Filter Banks """
        return self._data.shape[0]

    @property
    def filterSize(self) -> int:
        """ Return the size of each Mel Filter """
        return self._data.shape[1]

    def getEnergies(self) -> np.ndarray:
        """ Return the raw MFBE array """
        return self._data

    def getMeans(self) -> np.ndarray:
        """ Return mean energy of each filter """
        return np.mean(self._data,axis=0)

    def getVariances(self) -> np.ndarray:
        """ Return variance of energy in each filter """
        return np.var(self._data,axis=0)

    def getMedian(self) -> np.ndarray:
        """ Return the median energy of each filter bank """
        return np.median(self._data,axis=0)

    def getMin(self) -> np.ndarray:
        """ Return the minimum energy of each filter bank """
        return np.min(self._data,axis=0)

    def getMax(self) -> np.ndarray:
        """ Return the maximim energy of each filter bank """
        return np.max(self._data,axis=0)

    # Private Interface

    def __createCepstralCoeffs(self):
        """ Create Mel Freq Cepstral Coeffs from Filter bank energies """
        # TODO: Populate MFCCs
        return None

    # Magic Methods

    def __getitem__(self,index) -> object:
        """ Return item at index """
        return self._data[index]