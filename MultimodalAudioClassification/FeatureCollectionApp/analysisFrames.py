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
                 samplesPerFrame=2048,
                 sampleOverlap=512,
                 headPad=2048,
                 tailPad=4096,
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
        return np.sum(self.getFreqAxisMask())

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
        mask = ((freqAxis >= self.freqLowBoundHz) & (freqAxis < self.freqHighBoundHz))
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
        mask = ((freqAxis >= self.freqLowBoundHz) & (freqAxis < self.freqHighBoundHz))
        return freqAxis[mask]

    def getMelFilters(self,numFilters: int, normalize: bool) -> np.ndarray:
        """ Return the Mel Filter banks """
        if (self._melFilters.get(numFilters,None) is None):
            self._melFilters[numFilters] = self.__createMelFilters(numFilters)
        if (normalize == True):
            for ii in range(numFilters):
                filterArray = self._melFilters[numFilters][ii]
                filterArray /= np.sum(filterArray)
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
        result = self._data
        if (onlyInUse == True):
            result = self._data[0:self._framesInUse]
        return result

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
                 frameParams: AnalysisFrameParameters,
                 allowIncompleteFrames=False):
        """ Constructor """
        super().__init__(   signalData,
                            frameParams,
                            frameParams.maxNumFrames,
                            frameParams.getTimeFrameSize(),
                            TimeSeriesAnalysisFrames.__DATA_TYPE)
        self._allowIncompleteFrames = allowIncompleteFrames
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
                if (self._allowIncompleteFrames == True):
                    if (frameStart > len(signalData)):
                        break
                    frameEnd = len(signalData) - 1
                    frameSlice = signalData[frameStart:frameEnd]
                    self._data[ii,0:frameSlice.size] = frameSlice
                else:
                    continue
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

        #self.showHeatmap()
        
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

    def showHeatmap(self):
        """ Create a heatmap plot to show each frame """
        plt.figure(figsize=(16,12),facecolor="gray")
        plt.title("Frequency-Series Analysis Frames",size=32,weight="bold")
        plt.ylabel("Frame Index",size=24,weight="bold")
        plt.xlabel("Linear Frequency Bin",size=24,weight="bold")

        plt.pcolormesh(np.log(np.abs(self._data)**2),cmap="jet")

        plt.legend()
        plt.grid(True)
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
        if (signalData.cachedData.analysisFramesTime.getParams() != self._params):
            errMsg = "Provided signal's time-series analysis frames parmas do NOT match this one's"
            raise RuntimeWarning(errMsg)
        return True

    def _populateFrames(self,
                  signalData) -> None:
        """ OVERRIDE: Populate the analysis frames """
        if (self.useMultipleThreads == True):
            self.__populateWithMultipleThreads(signalData)
        else:
            self.__populateWithSingleThread(signalData)
        return None

    # Private Interface

    def __populateWithMultipleThreads(self,
                                     signalData) -> None:
        """ Populate frequency series analysis frames in multiple threads """
        # TODO: Implement this later
        return None

    def __populateWithSingleThread(self,
                                   signalData) -> None:
        """ Populate frequency Series analysis frames in a single thread """
        #freqAxis = self._params.getFreqAxisMasked()
        freqAxis = self._params.getFreqAxisMasked()
        for ii in range(self._framesInUse):
            self._data[ii] = self.__transform(signalData.cachedData.analysisFramesTime[ii])
            #debugPlotXy(freqAxis,self._data[ii],"Frame{0}".format(ii))
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
        self._filterMatrix  = self._params.getMelFilters(numFilters,True)
        self._data          = None

        self.__validateSignal(signal)

        numFrames = signal.cachedData.analysisFramesFreq.getNumFramesInUse()
        self._data = np.zeros(shape=(numFrames,numFilters),dtype=np.float32)

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
    def numFrames(self) -> int:
        """ Return the number of Mel Filter Banks """
        return self._data.shape[0]

    @property
    def numFilters(self) -> int:
        """ Return the size of each Mel Filter """
        return self._data.shape[1]

    def getEnergies(self) -> np.ndarray:
        """ Return the raw MFBE array """
        return self._data

    def getMeans(self,normalize=False) -> np.ndarray:
        """ Return mean energy of each filter """
        means = np.mean(self._data,axis=0)
        if (normalize == True):
            means /= np.max(means)
        return means

    def getVariances(self,normalize=False) -> np.ndarray:
        """ Return variance of energy in each filter """
        varis = np.var(self._data,axis=0)
        if (normalize == True):
            varis /= np.max(varis)
        return varis

    def getMedians(self,normalize=False) -> np.ndarray:
        """ Return the median energy of each filter bank """
        medis = np.median(self._data,axis=0)
        if (normalize == True):
            medis /= np.max(medis)
        return medis

    def getMins(self) -> np.ndarray:
        """ Return the minimum energy of each filter bank """
        return np.min(self._data,axis=0)

    def getMaxes(self) -> np.ndarray:
        """ Return the maximim energy of each filter bank """
        return np.max(self._data,axis=0)

    # Public Interface

    def plotMelFilters(self):
        """ Create a plot to show shape of each mel filter """
        plt.figure(figsize=(16,12),facecolor="gray")
        plt.title("Mel Filters",size=32,weight="bold")
        plt.xlabel("Frequency [hz]",size=24,weight="bold")
        plt.ylabel("Filter Level",size=24,weight="bold")

        for ii in range(self.numFilters):
            label = "Mel Filter #{0}".format(ii)
            plt.plot(self._filterMatrix[ii],label=label)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return None

    def plotEnergiesByFrame(self,plotLogScale=True):
        """ Create a plot to show the the energy of each filter bank changes by each frame """
        plt.figure(figsize=(16,12),facecolor="gray")
        plt.title("Mel Filter Bank Energies by Frame",size=32,weight="bold")
        plt.xlabel("Frame Index",size=24,weight="bold")
        plt.ylabel("Energy Level",size=24,weight="bold")
        
        for ii in range(self.numFilters):
            energyData = self._data[:,ii]
            if (plotLogScale == True):
                energyData = np.log10(energyData)
            label = "MFBE #{0}".format(ii)
            plt.plot(energyData,label=label)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return None

    # Private Interface

    def __validateSignal(self,
                        signalData) -> bool:
        """ Validate that the input signal has info to work with """
        if (signalData.cachedData.analysisFramesTime is None):
            errMsg = "Provided signal does not have time-series analysis frames"
            raise RuntimeWarning(errMsg)
        if (signalData.cachedData.analysisFramesTime.getParams() != self._params):
            errMsg = "Provided signal's analysis frames parmas do NOT match this one's"
            raise RuntimeWarning(errMsg)
        return True

    def __applyMelFilters(self,signal) -> np.ndarray:
        """ Apply mel Filters to freq-series frames """
        rawFrames = signal.cachedData.analysisFramesFreq.rawFrames(onlyInUse=True)
        freqFrames = np.abs(rawFrames)**2

        for ii in range(self.numFrames):
            for jj in range(self.numFilters):
                self._data[ii,jj] = np.dot(
                    freqFrames[ii],
                    self._filterMatrix[jj])
        #self.plotEnergiesByFrame(plotLogScale=True)
        return None

    # Magic Methods

    def __getitem__(self,index) -> object:
        """ Return item at index """
        return self._data[index]

class MelFrequencyCepstralCoefficients:
    """ Stores the Mel Frequency Cepstral Coefficients """

    def __init__(self,
                 signal,
                 frameParams,
                 numCoeffs: int):
        """ Constructor """
        self._params        = frameParams
        self._filterMatrix  = self._params.getMelFilters(numCoeffs,True)
        self._data          = None

        self.__validateSignal(signal)

        numFrames = signal.cachedData.analysisFramesFreq.getNumFramesInUse()
        self._data = np.zeros(shape=(numFrames,numCoeffs))

        self.__createCepstralCoeffs(signal)

    def __del__(self):
        """ Destructor """
        self._data = None

    # Accessors

    def getParams(self) -> AnalysisFrameParameters:
        """ Return the parameters structure used to create this instance """
        return self._params

    @property
    def numFrames(self) -> int:
        """ Return the number of Mel Filter Banks """
        return self._data.shape[0]

    @property
    def numCoeffs(self) -> int:
        """ Return the size of each Mel Filter """
        return self._data.shape[1]

    def getCoeffs(self) -> np.ndarray:
        """ Return the raw MFBE array """
        return self._data

    def getMeans(self) -> np.ndarray:
        """ Return mean energy of each filter """
        return np.mean(self._data,axis=0)

    def getVariances(self) -> np.ndarray:
        """ Return variance of energy in each filter """
        return np.var(self._data,axis=0)

    def getMedians(self) -> np.ndarray:
        """ Return the median energy of each filter bank """
        return np.median(self._data,axis=0)

    def getMin(self) -> np.ndarray:
        """ Return the minimum energy of each filter bank """
        return np.min(self._data,axis=0)

    def getMax(self) -> np.ndarray:
        """ Return the maximim energy of each filter bank """
        return np.max(self._data,axis=0)

    # Public Interface

    def plotCepstrumByFrame(self):
        """ Create a plot to show the the energy of each filter bank changes by each frame """
        plt.figure(figsize=(16,12),facecolor="gray")
        plt.title("Mel Frequency Cepstral Coefficients by Frame",size=32,weight="bold")
        plt.xlabel("Frame Index",size=24,weight="bold")
        plt.ylabel("Energy Level",size=24,weight="bold")
        
        for ii in range(self.numCoeffs):
            energyData = self._data[:,ii]
            label = "MFCC #{0}".format(ii)
            plt.plot(energyData,label=label)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return None

    def showHeatmap(self):
        """ Create a heatmap plot to show each frame """
        plt.figure(figsize=(16,12),facecolor="gray")
        plt.title("Mel Frequency Cepstral Coefficients",size=32,weight="bold")
        plt.ylabel("Frame Index",size=24,weight="bold")
        plt.xlabel("MFCC Bin",size=24,weight="bold")

        plt.pcolormesh(self._data,cmap="jet")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return None

    # Private Interface

    def __validateSignal(self,
                        signalData) -> bool:
        """ Validate that the input signal has info to work with """
        if (signalData.cachedData.melFilterFrameEnergies is None):
            errMsg = "Provided signal does not have mel filter bank energies"
            raise RuntimeWarning(errMsg)
        if (signalData.cachedData.melFilterFrameEnergies.getParams() != self._params):
            errMsg = "Provided signal's mel filter bank energies params do NOT match this one's"
            raise RuntimeWarning(errMsg)
        return True

    def __createCepstralCoeffs(self,signal):
        """ Create Mel Freq Cepstral Coeffs from Filter bank energies """
        logEnergies = np.log10(signal.cachedData.melFilterFrameEnergies.getEnergies())
        for tt in range(self.numFrames):
            for cc in range(self.numCoeffs):
                for mm in range (self.numCoeffs):
                    cosTerm = np.cos((cc + 1) * (mm + 0.5) * np.pi / self.numCoeffs)
                    self._data[tt,cc] += logEnergies[tt,mm] * cosTerm
        self._data /= np.sqrt(2 / self.numCoeffs)
        self._data /= np.max(np.abs(self._data))
        return None

    # Magic Methods

    def __getitem__(self,index) -> object:
        """ Return item at index """
        return self._data[index]