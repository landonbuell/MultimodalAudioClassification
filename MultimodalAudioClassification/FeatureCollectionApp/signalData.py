"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       signalData.py
    Classes:    SignalData, 
                SignalData.CachedData

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import os
import numpy as np
import matplotlib.pyplot as plt

import analysisFrames

        #### CLASS DEFINITIONS ####

class SignalData:
    """ Stores information related to a single Waveform """

    class CachedData:
        """ Stores Additional Signal Data information rather than recompute it """

        def __init__(self):
            """ Constructor """
            self.analysisFramesTime = None
            self.analysisFramesFreq = None
            self.freqCenterOfMasses = None
            self.melFilterFrameEnergies  = None
            self.melFreqCepstralCoeffs = None

        def __del__(self):
            """ Destructor """
            pass

    # Applies a unique label to each sample
    __unqiueCounter = 0

    # Constructors 

    def __init__(self,
                 sampleRate=44100,
                 targetClass=-1,
                 waveform=None,
                 sourcePath="NULL_PATH",
                 channelIndex=0):
        """ Constructor """
        self._sampleRate    = sampleRate
        self._targetClass   = targetClass
        self._waveform      = waveform
        self._cachedData    = SignalData.CachedData()
        self._sourcePath    = sourcePath
        self._channelIndex  = channelIndex
        self._uniqueID      = SignalData.__unqiueCounter

        SignalData.__unqiueCounter += 1
        if (self._waveform is None):
            self._waveform = np.array(shape=(1,),dtype=np.float32)

    def __del__(self):
        """ Destructor """
        self._cachedData.clear()

    # Accessors

    def getSampleRate(self) -> int:
        """ Return the sample rate for this signal """
        return self._sampleRate

    def setSampleRate(self,
                      sampleRate: int) -> None:
        """ Set the sample rate for this signal """
        self._sampleRate = sampleRate
        return None

    def getTarget(self) -> int:
        """ Return the target class for this signal """
        return self._targetClass

    def setTarget(self,
                  targetClass: int) -> None:
        """ Set the tatget class for this signal """
        self._targetClass = targetClass
        return None

    def getWaveform(self) -> np.ndarray:
        """ Return the waveform """
        return self._waveform

    def setWaveform(self,
                    waveform: np.ndarray) -> None:
        """ Set the waveform """
        self._waveform = waveform
        return None

    @property
    def waveform(self):
        """ Return the waveform as a property """
        return self._waveform

    @property
    def cachedData(self) -> CachedData:
        """ Return the underlying Cached data for this signal """
        return self._cachedData

    def getSourcePath(self) -> str:
        """ Return the path that this signal originated from """
        return self._sourcePath

    def setSourcePath(self,
                      sourcePath: str) -> None:
        """ Set the path that this signal originated from """
        self._sourcePath = sourcePath
        return None

    def getChannelIndex(self) -> int:
        """ Return the channel index that this signal originted from """
        return self._channelIndex

    def setChannelIndex(self,
                        channelIndex: int) -> None:
        """ Set the channel index that this signal originated from """
        self._channelIndex = channelIndex
        return None

    def getNumSamples(self) -> int:
        """ Return the number of samples in this waveform """
        return self._waveform.size

    @property
    def numSamples(self) -> int:
        """ Return the number of samples in this waveform """
        return self._waveform.size

    def uniqueID(self) -> int:
        """ Return the unique counter ID for this signal """
        return self._uniqueID

    # Public Interface

    def exportNameBinary(self) -> str:
        """ Name of the exported binary file """
        return "sample{0}.bin".format(self.uniqueID())

    def clearCachedData(self) -> None:
        """ Clear the underlying cached data """
        # TODO: Implement this!
        return None

    def normalizeAmplitude(self,newDataType=np.float32) -> None:
        """ Normalize ampltiude to +/- 1. Recast to type if applicable """
        newWaveform = self.waveform.astype(newDataType)
        newWaveform -= np.mean(newWaveform)
        newWaveform /= np.max(np.abs(newWaveform))
        self._waveform = newWaveform
        return None

    def show(self) -> None:
        """ Plot the time-series representation of this waveform """
        titleText = "{0} \n ch#{1}".format(self.getSourcePath(),str(self.getChannelIndex()))
        
        plt.figure(figsize=(16,12))
        plt.title(titleText,size=24,weight='bold')
        plt.xlabel("Time [Sample Index]",size=16,weight='bold')
        plt.ylabel("Amplitude",size=16,weight='bold')

        waveformSlice = self.waveform[:int(2**12)]
        plt.plot(waveformSlice,color='blue')

        plt.vlines(0,ymin=np.min(waveformSlice),ymax=np.max(waveformSlice),color='black')
        plt.hlines(0,0,len(waveformSlice),color='black')

        plt.grid()
        plt.tight_layout()

        plt.show()
        return None

    def makeTimeSeriesAnalysisFrames(self,
                                     frameParams: analysisFrames.AnalysisFrameParameters,
                                     forceMakeFrames=False) -> bool:
        """ Populate the cached data' time series analysis frames """
        if (forceMakeFrames == True):
            self.cachedData.analysisFramesTime = analysisFrames.TimeSeriesAnalysisFrames(self,frameParams)
            return True
        if (self.__shouldMakeTimeSeriesAnalysisFrames(frameParams) == True):
            self.cachedData.analysisFramesTime = analysisFrames.TimeSeriesAnalysisFrames(self,frameParams)
            return True
        return False

    def makeFreqSeriesAnalysisFrames(self,
                                     frameParams: analysisFrames.AnalysisFrameParameters,
                                     forceMakeFrames=False) -> bool:
        """ Populate the cached data' frequency series analysis frames """
        self.makeTimeSeriesAnalysisFrames(frameParams,forceMakeFrames)
        if (forceMakeFrames == True):
            self.cachedData.analysisFramesFreq = analysisFrames.FreqSeriesAnalysisFrames(self,frameParams)
            return True
        if (self.__shouldMakeFreqSeriesAnalysisFrames(frameParams) == True):
            self.cachedData.analysisFramesFreq = analysisFrames.FreqSeriesAnalysisFrames(self,frameParams)
            return True
        return False

    def makeFreqCenterOfMasses(self,
                                  frameParams: analysisFrames.AnalysisFrameParameters,
                                  forceMakeFrames=False) -> bool:
        """ Populate the cached data' frequency center of mass frames """
        madeFrames = self.makeFreqSeriesAnalysisFrames(frameParams,forceMakeFrames)
        weights = np.arange(0,frameParams.freqFrameSize,1)
        nmrtr = np.dot(self.cachedData.analysisFramesFreq.getRawFrames(),weights)
        dnmtr = np.sum(np.abs(self.cachedData.analysisFramesFreq.getRawFrames()),axis=1) + 1e-8
        self.cachedData.freqCenterOfMasses = (nmrtr / dnmtr)
        if (self.cachedData.freqCenterOfMasses.size != frameParams.maxNumFrames):
            msg = "Mismatch in number of frames & number of center of Masses"
            raise RuntimeError(msg)
        return madeFrames

    def makeMelFilterBankEnergies(self,
                                  numFilters: int,
                                  frameParams: analysisFrames.AnalysisFrameParameters,
                                  forceMakeFrames=False,) -> bool:
        """ Populate the cached data Mel Filter Bank Energies """
        self.makeFreqSeriesAnalysisFrames(frameParams,forceMakeFrames) 
        if (forceMakeFrames == True):
            self.cachedData.melFilterFrameEnergies = analysisFrames.MelFilterBankEnergies(self,frameParams,numFilters)
            return True
        if (self.__shouldMakeMelFilterBankEnergies(frameParams) == True):
            self.cachedData.melFilterFrameEnergies = analysisFrames.MelFilterBankEnergies(self,frameParams,numFilters)
            return True
        return False

    def makeMelFreqenctCepstralCoeffs(self,
                                    numFilters: int,
                                    frameParams: analysisFrames.AnalysisFrameParameters,
                                    forceMakeFrames=False,) -> bool:
        """ Populate the MFCCs """
        return False


    # Private Interface

    def __shouldMakeTimeSeriesAnalysisFrames(self,
                                            frameParams: analysisFrames.AnalysisFrameParameters) -> bool:
        """ Return T/F if we should make or remake analysis time-series analysis frames based on provided params """
        if (self.cachedData.analysisFramesTime is None):
            # Frames do not exist, we should make them
            return True
        if (self.cachedData.analysisFramesTime.getParms() != frameParams):
            # The provided params do not match to existing params
            return True
        return False

    def __shouldMakeFreqSeriesAnalysisFrames(self,
                                            analysisFrameParams: analysisFrames.AnalysisFrameParameters) -> bool:
        """ Return T/F if we should make or remake analysis time-series analysis frames based on provided params """
        if (self.cachedData.analysisFramesFreq is None):
            # Frames do not exist, we should make them
            return True
        if (self.cachedData.analysisFramesFreq.getParams() != analysisFrameParams):
            # The provided params do not match to existing params
            return True
        return False

    def __shouldMakeMelFilterBankEnergies(self,
                                          analysisFrameParams: analysisFrames.AnalysisFrameParameters) -> bool:
        """ Return T/F id we should make or remake the mel-filter bank energies based on provided params """
        
        if (self.cachedData.melFilterFrameEnergies is None):
            # MFBEs do not exist, so we should make them
            return True
        if (self.cachedData.melFilterFrameEnergies.getParams() != analysisFrameParams):
            # The provided params to not match the existing ones
            return True
        return False


    # Magic Methods

    def __len__(self) -> int:
        """ Return the number of samples in the waveform """
        return self._waveform.size

    def __getitem__(self,
                    key: int) -> np.ndarray:
        """ Return the frame at the provided value """
        return self._waveform[key]

    def __setitem__(self,
                    key: int,
                    val: np.ndarray) -> None:
        """ Set the provided value at the provided frame key """
        self._waveform[key] = val
        return None

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))

