"""
Repository:     MultimodalAudioClassification
Solution:       MultimodalAudioClassification
Project:        FeatureCollection  
File:           Administrative.py
 
Author:         Landon Buell
Date:           December 2021
"""

        #### IMPORTS ####

import os
import numpy as np
import matplotlib.pyplot as plt

import scipy.io.wavfile as sciowav
import scipy.fftpack as fftpack
import scipy.signal as scisig

import Administrative
import Managers
import CollectionMethods

        #### CLASS DEFINITIONS ####

class SampleIO:
    """ Sample IO Contains Data for Each Audio File to Read """

    def __init__(self,path,targetInt,targetStr=None):
        """ Constructor for SampleIO Instance """
        self._filePath      = path
        self._targetInt     = targetInt
        self._targetStr     = targetStr
        self._reqSamples    = int(2**18)

    def __del__(self):
        """ Destructor for SampleIO Instance """
        pass

    # Getters and Setters

    def getFilePath(self) -> str:
        """ Return the File Path """
        return self._filePath

    def getTargetInt(self) -> int:
        """ Return Target Label as Int """
        return self._targetInt

    def getTargetStr(self) -> str:
        """ Return Target Label as Str """
        return self._targetStr

    def getPathExtension(self) -> str:
        """ Get File Type Ext """
        return self._filePath.split(".")[-1]

    # Public Interface

    def readSignal(self):
        """ Read The Audio From the indicate file """
        ext = self.getPathExtension()
        if (ext == "wav"):
            # Wav File
            return self.__readFileWav()
        else:
            # Not Implements
            errMsg = "File extension: " + ext + " is not yet supported"
            raise RuntimeError(errMsg)

    # Private Interface

    def __readFileWav(self):
        """ Read Data From .wav file """
        sampleRate,data = sciowav.read(self._filePath)
        waveform = data.astype(dtype=np.float32).flatten()
        waveform -= np.mean(waveform)
        waveform /= np.max(np.abs(waveform))       
        waveform = self.__padWaveform(waveform)    
        return SignalData(waveform,self._targetInt,self._targetStr,sampleRate)

    def __padWaveform(self,waveform):
        """ Pad or Crop Waveform if too long or too short """
        if (waveform.shape[0] < self._reqSamples):
            # Too few samples
            deficit = self._reqSamples - waveform.shape[0]
            waveform = np.append(waveform,np.zeros(shape=deficit,dtype=np.float32))
        elif (waveform.shape[0] > self._reqSamples):
            # Too many samples
            waveform = waveform[0:self._reqSamples]
        else:
            # Exactly the right number of samples - do nothing
            pass
        return waveform

    # Magic Methods
    
    def __repr__(self) -> str:
        """ Debug representation of instance """
        return "Sample: " + self.getPathExtension() + " " + self.getTargetStr()

class SignalData:
    """ Contain all signal Data (NOT ENCAPSULATED) """

    def __init__(self,waveform,targetInt,targetStr,sampleRate=44100):
        """ Constructor for SignalData Instance """
        self._sampleRate            = sampleRate
        self._targetInt             = targetInt
        self._targetStr             = targetStr
        self.Waveform               = waveform
        self.AnalysisFramesTime     = None
        self.AnalysisFramesFreq     = None
        self.FreqCenterOfMasses     = None
        self.MelFilterBankEnergies  = None
        self.AutoCorrelationCoeffs  = None
        self.FrameZeroCrossings     = None
        self.FrameEnergyTime        = None
        self.FrequencyAxis          = None

    def __del__(self):
        """ Destructor for SignalData Instance """
        self.clear()
       
    # Getters and Setters

    def getSampleRate(self):
        """ Get the Sample Rate """
        return self._sampleRate

    def setSampleRate(self,rate):
        """ Set the Sample Rate """
        self._sampleRate = rate
        return self

    def getTargetInt(self) -> int:
        """ Return Target Label as Int """
        return self._targetInt

    def getTargetStr(self) -> str:
        """ Return Target Label as Str """
        return self._targetStr

    def getWaveform(self):
        """ Get the Signal Samples """
        return self.Waveform

    def setWaveform(self,data):
        """ Set the Signal Samples """
        self.clear()
        self.Waveform = data
        return self

    def getNumSamples(self):
        """ Get the Number of samples in the waveform """
        return self.Waveform.shape[0]

    def getSampleSpace(self):
        """ Get the Sample Spacing """
        return (1/self._sampleRate)

    def getNumAnalysisFramesTime(self):
        """ Get the Number of Time Series analysis frames """
        if (self.AnalysisFramesTime is None):
            return 0
        else:
            return self.AnalysisFramesTime.shape[0]

    def getNumAnalysisFramesFreq(self):
        """ Get the Number of Time Series analysis frames """
        if (self.AnalysisFramesFreq is None):
            return 0
        else:
            return self.AnalysisFramesFreq.shape[0]

    # Public Interface

    def clear(self):
        """ Clear all Fields of the Instance """
        self.AnalysisFramesTime     = None
        self.AnalysisFramesFreq     = None
        self.FreqCenterOfMasses     = None
        self.MelFilterBankEnergies  = None
        self.AutoCorrelationCoeffs  = None
        self.FrameZeroCrossings     = None
        self.FrameEnergyTime        = None
        return self

    def makeAnalysisFramesTime(self,frameParams=None):
        """ Build Time-Series AnalysisFrames """
        if (self.Waveform is None):
            # No Signal - Cannot Make Frames
            errMsg = "ERROR: need signal to make analysis Frames"
            raise RuntimeError(errMsg)

        # Create the Frames Constructor with Params
        if (frameParams is None):
            msg = "Cannot make TimeSeriesAnalysisFrames w/ NONE frame params"
            raise RuntimeError(msg)
        constructor = AnalysisFramesTimeConstructor(frameParams)
        constructor.call(self)
        constructor = None
        return self

    def makeAnalysisFramesFreq(self,frameParams=None):
        """ Make Frequncy-Series Analysis Frames """
        if (self.AnalysisFramesTime is None):
            # No Time  Analysis Frames - Make them
            self.makeAnalysisFramesTime(frameParams)
        # Apply window Function + Fourier Transform
        constructor = AnalysisFramesFreqConstructor(frameParams)
        constructor.call(self)
        return self

    def makeFrequencyCenterOfMass(self,weights):
        """ Compute Frequency Center of Mass for Each Analysis Frame """
        if (self.AnalysisFramesFreq is None):
            # No Freq Analysis Frames - Cannot make FCM's
            errMsg = "ERROR: need analysis frames time to make analysis frames frequency"
            raise RuntimeError(errMsg)
        # Compute Total "Mass"
        massTotals = np.sum(self.AnalysisFramesFreq,axis=-1) + CollectionMethods.EPSILON     
        # Compute Center of Mass (by Weights)
        massCenters = np.matmul(self.AnalysisFramesFreq,weights)
        massCenters /= massTotals
        self.FreqCenterOfMasses = massCenters
        return self

    def makeMelFilterBankEnergies(self,frameParams,numCoeffs):
        """ Make All Mel-Cepstrum Frequency Coefficients """
        if (self.AnalysisFramesFreq is None):
            # No Freq Analysis Frames - Cannot make MFCC's
            errMsg = "ERROR: need analysis frames time to make analysis frames frequency"
            raise RuntimeError(errMsg)

        # Create + Call the MFCC builder
        constructor = MelFrequnecyCepstrumCoeffsConstructor(
            frameParams,numCoeffs)
        constructor.call(self)
        return self

    def makeAutoCorrelationCoeffs(self,numCoeffs):
        """ Make All Auto-Correlation Coefficients """
        if (self.Waveform is None):
            # No Waveform - Cannot make ACC's
            errMsg = "ERROR: need analysis frames time to make analysis frames frequency"
            raise RuntimeError(errMsg)

        # Make the auto-correlation Coeffs
        self.AutoCorrelationCoeffs = np.zeros(shape=(numCoeffs,),dtype=np.float32)
        for k in range(1,numCoeffs+1,1):
            # Each ACC
            alpha = self.Waveform[0:-k]
            beta = self.Waveform[k:]
            sumA = np.dot(alpha,beta)
            sumB = np.dot(alpha,alpha)
            sumC = np.dot(beta,beta)
            acc = sumA / (np.sqrt(sumB) * np.sqrt(sumC))
            self.AutoCorrelationCoeffs[k - 1] = acc

        return self

    def makeZeroCrossingRate(self):
        """ Make Zero Crossing Rate of Each Frame """
        if (self.AnalysisFramesTime is None):
            # No Analysis Frames - Raise Error
            errMsg = "SignalData.makeZeroCrossingRate() - Need Time Series Analysis Frames"
            raise RuntimeWarning(errMsg)
        # Compute Sign of Each Element
        signs = np.sign(self.AnalysisFramesTime)
        numRows = self.AnalysisFramesTime.shape[0]
        numCols = self.AnalysisFramesTime.shape[1]
        result = np.zeros(shape=(numRows,),dtype=np.float32)

        # Compute Zero Crossings Within Each Frame
        for i in range(numRows):
            
            # ZXR of this Frame
            zxr = 0
            for j in range(1,numCols):
                 zxr += np.abs(signs[i,j] - signs[i,j-1])
            result[i] = zxr

        # Attach Result to Self
        self.FrameZeroCrossings = result
        return self

    def makeFrameEnergiesTime(self):
        """ Make Time-Series Energiees for Each Analysis Frame """
        if (self.AnalysisFramesTime is None):
            # No Analysis Frames - Cannot Make Energies
            errMsg = "ERROR: need analysis frames to make analysis Frames"
            raise RuntimeError(errMsg)
        result = np.sum(self.AnalysisFramesTime**2,axis=-1,dtype=np.float32)
        self.FrameEnergyTime = result
        return self

    # Plotting Helpers

    # Private Interface

    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class__) + " " + self._className + " @ " + str(hex(id(self)))

class AnalysisFramesParameters:
    """ AnalysisFramesParamaters contains 
    values to use when building Analysis Frames """

    def __init__(self,samplesPerFrame=1024,samplesOverlap=768,
                 headPad=1024,tailPad=2048,maxFrames=256,
                 window="hanning",freqLowHz=0,freqHighHz=12000,
                 sampleRate=44100):
        """ Constructor for AnalysisFramesParameters Instance """

        # For Time Series Frames
        self._samplesPerFrame   = samplesPerFrame
        self._samplesOverlap    = samplesOverlap
        self._padHead           = headPad
        self._padTail           = tailPad
        self._maxFrames         = maxFrames
        self._framesInUse       = 0

        # For Frequency Series Frames
        self._windowFunction    = window
        self._freqLowHz         = freqLowHz
        self._freqHighHz        = freqHighHz
        self._sampleRate        = sampleRate

    def __del__(self):
        """ Destructor for AnalysisFramesParameters Instance """
        pass

    def reset(self):
        """ Clear the State of the Instance, reset to Construction """
        self._framesInUse       = 0
        return self

    # Getters and Setters

    def getMaxNumFrames(self) -> int:
        """ Get the Max Number of Frames to Use """
        return self._maxFrames

    def getNumFramesInUse(self) -> int:
        """ Get the Number of Frames Currently in use """
        return self._framesInUse

    def getFreqBoundHz(self):
        """ Get the Low + High Freq Bound in Hz """
        return np.array([self._freqLowHz, self._freqHighHz])

    def getTotalTimeFrameSize(self) -> int:
        """ Get total Size of Each Time Frame including padding """
        result = 0
        result += self._padHead
        result += self._samplesPerFrame 
        result += self._padTail
        return result

    def getTotalFreqFrameSize(self):
        """ Get total Size of Each Frequency Frame including padding """
        arr = self.generateFreqAxis()
        size = arr.shape[0]
        return size

    def getTimeFramesShape(self):
        """ Get the Shape of the Time-Series Analysis Frames Matrix """
        return ( self.getMaxNumFrames(), self.getTotalTimeFrameSize(), )

    def getFreqFramesShape(self,sampleRate=44100):
        """ Get the Shape of the Freq-Series Analysis Frames Matrix """
        return (self.getMaxNumFrames(), self.getTotalFreqFrameSize(), )

    # Public Interface

    def generateTimeAxis(self):
        """ Generate an array that represents the time axis """
        arr = np.arange(0,self._maxFrames,1,dtype=np.float64)
        secondsPerFrameOverlap = self._samplesOverlap * (1/self._sampleRate)
        arr *= secondsPerFrameOverlap
        return arr

    def generateFreqAxis(self):
        """ Generate an array that reprents the frequency axis """
        fftAxis = fftpack.fftfreq(self.getTotalTimeFrameSize(),1/self._sampleRate)
        mask = np.where(
            (fftAxis>=self._freqLowHz) & 
            (fftAxis<=self._freqHighHz) )[0]   # get slices
        return fftAxis[mask]

    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

class AnalysisFramesConstructor:
    """ Abstract Base Class for Construting Time/Freq Analysis Frames """

    def __init__(self,frameParams):
        """ Constructor for AnalysisFramesConstructor Base Class """
        self._params    = frameParams
        self._signal    = None

    def __del__(self):
        """ Destructor for AnalysisFramesConstructor Instance """
        self._params    = None
        self._signal    = None

    # Getters and Setters

    def getSamplesPerFrame(self) -> int:
        """ Get the Number of Samples in Each Frame """
        return self._params._samplesPerFrame

    def getSamplesOverlap(self) -> int:
        """ Get the Number of Overlap Samples in Each Frame """
        return self._params._samplesOverlap

    def getSizeHeadPad(self) -> int:
        """ Get the Size of the Head Pad """
        return self._params._padHead

    def getSizeTailPad(self) -> int:
        """ Get the size of the Tail Pad """
        return self._params._padTail

    def getMaxNumFrames(self) -> int:
        """ Get the Max Number of Frames to Use """
        return self._params._maxFrames

    def getNumFramesInUse(self) -> int:
        """ Get the Number of Frames Currently in use """
        return self._params._framesInUse

    def getSampleStep(self) -> int:
        """ Get the Sample Step Between adjacent analysis frames """
        return (self._params._samplesPerFrame - self._params._samplesOverlap)

    def getTotalFrameSize(self) -> int:
        """ Get total Size of Each Frame including padding """
        result = 0
        result += self._params._padHead
        result += self._params._samplesPerFrame 
        result += self._params._padTail
        return result

    def getFramesShape(self):
        """ Get the Shape of the Analysis Frames Matrix """
        return ( self.getMaxNumFrames(), self.getTotalFrameSize(), )
    
    # Public Interface

    def emptyFrames(self):
        """ Return Empty 2D array that frames will occupy """
        return np.zeros(shape=self.getFramesShape(),dtype=np.float32)

    def call(self,signalData):
        """ Run Frames Constructor Instance w/ signal Data instance """
       
        # Attach a refrence of the signal to self (save typing)
        self._signal = signalData
        return signalData

    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

class AnalysisFramesTimeConstructor(AnalysisFramesConstructor):
    """ FrameParameters Structure to Create Freq-Series AnalysisFrames """

    def __init__(self,frameParams):
        """ Constructor for AnalysisFramesTimeConstructor Instance """
        super().__init__(frameParams)

    def __del__(self):
        """ Destructor for AnalysisFramesTimeConstructor Instance """
        super().__del__()

    # Public Interface
    
    def call(self,signalData):
        """ Convert Signal to Analysis Frames """
        super().call(signalData)
        self._signal.AnalysisFramesTime = self.emptyFrames()
        self.__buildFramesTime()
        self._signal = None         # remove refrence
        # Return the New Signal Data Object
        return signalData

    # Private Interface

    def __buildFramesTime(self):
        """ Construct Analysis Time-Series Frames """
        startIndex = 0
        numSamples = self._signal.Waveform.shape[0]
        padHead = self.getSizeHeadPad()
        frameSize = self.getSamplesPerFrame()

        # Copy all of the frames
        for i in range(self.getMaxNumFrames()):
        
            # Copy slice to padded row
            np.copyto(
                dst=self._signal.AnalysisFramesTime[i,padHead:padHead + frameSize],
                src=self._signal.Waveform[startIndex:startIndex + frameSize],
                casting='no')
            
            # Increment
            startIndex += self.getSampleStep()
            self._params._framesInUse += 1

            if (startIndex + frameSize > numSamples):
                # That was just the last frame
                break
        
        if (self.getNumFramesInUse() < self.getMaxNumFrames()):
            # Crop Unused Frames
            self._signal.AnalysisFramesTime = \
                self._signal.AnalysisFramesTime[0:self._params._framesInUse]
        # Return Instance
        return self

class AnalysisFramesFreqConstructor(AnalysisFramesConstructor):
    """ Create Freq-Series AnalysisFrames """

    def __init__(self,frameParams):
        """ Constructor for AnalysisFramesFreqConstructor Instance """
        super().__init__(frameParams)

    def __del__(self):
        """ Destructor for AnalysisFramesFreqConstructor Instance """
        super().__del__()

    # Public Interface 

    def call(self,signalData):
        """ Convert Signal to Analysis Frames """
        super().call(signalData)
        if (signalData.AnalysisFramesTime is None):
            # Must have time-Frames
            errMsg = "AnalysisFramesFreqConstructor.call() - Must have Time-Frames to make Freq-Frames"
            raise RuntimeError(errMsg)
        self.__buildFramesFreq()
        self._signal = None         # remove refrence
        # Return the New Signal Data Object
        return signalData

    # Private Interface

    def __buildFramesFreq(self):
        """ Construct Analysis Time-Series Frames """
        # Get the winow and apply to each frame
        window = WindowFunctions.getHanning(
            self.getSamplesPerFrame(),self.getSizeHeadPad(),self.getSizeTailPad())
        frames = self._signal.AnalysisFramesTime * window

        # Apply the DFT to the frames matrix
        frames = fftpack.fft(frames,axis=-1,)
        frames = frames / frames.shape[1]
        frames = np.abs(frames,dtype=np.float32)**2

        # Crop the Frames to the Frequency Spectrum subset
        freqAxis,mask = self.__frequencyAxis(self._signal.getSampleSpace())
        frames = frames[:,mask];
        #timeAxis = np.arange(0,self.getMaxNumFrames() )
        #PyToolsPlotting.spectrogram(frames,timeAxis,freqAxis,"Spectrogram")

        self._signal.AnalysisFramesFreq = frames
        self._signal.FrequencyAxis = freqAxis
        return self

    def __frequencyAxis(self,sampleSpacing=1):
        """ Get the Frequnecy-Space Axis for the Analysis Frames """
        space = fftpack.fftfreq(self.getTotalFrameSize(),sampleSpacing)
        mask = np.where(
            (space>=self._params._freqLowHz) & 
            (space<=self._params._freqHighHz) )[0]   # get slices
        space = space[mask]
        return space,mask

class MelFrequnecyCepstrumCoeffsConstructor:
    """ Class the Handle the Creation of all Mel-Frequency-Cepstrum Coeffs """

    def __init__(self,frameParams,numCoeffs,freqLowHz=0,freqHighHz=22050,sampleRate=44100):
        """ Constructor for MelFrequnecyCepstrumCoeffsConstructor Instance """
        self._numCoeffs = numCoeffs
        self._freqLowHz = freqLowHz
        self._freqHighHz = freqHighHz
        self._sampleRate = sampleRate
        self._melFilterBanks = self.__buildMelFilterBanks(frameParams)
       

    def __del__(self):
        """ Destructor for MelFrequnecyCepstrumCoeffsConstructor Instance """
        self._numCoeffs = 0
        self._signal = None

    def call(self,signalData):
        """ Create Mel-Freqency Cepstrum Coeffs from Analysis Frames """
        signalData.MelFilterBankEnergies = np.empty(
            shape=(signalData.AnalysisFramesFreq.shape[0],self._numCoeffs,),
            dtype=np.float32)
        # Compute the MFCCs for Each Freq-Series Analysis Frames
        np.matmul(  signalData.AnalysisFramesFreq,
                    self._melFilterBanks.transpose(),       
                    out=signalData.MelFilterBankEnergies)        
        return signalData

    # Private Interface

    def __buildMelFilterBanks(self,frameParams):
        """ Construct the Mel Filter Bank Envelopes """
        filters = CollectionMethods.MelFilterBankEnergies.melFilters(
            frameParams,self._numCoeffs,self._sampleRate)
        return filters

class WindowFunctions:
    """ Static Class to Hold All Window Functions """

    PadHead = 1024
    PadTail = 2048
    windowSize = lambda x,y,z : x + y + z

    def __init__(self):
        """ Dummy Constructor - Raises Error """
        errMsg = str(self.__class__) + " is a static class, cannot make instance"
        raise RuntimeError(errMsg)

    @staticmethod
    def getWindowSize(*items):
        """ Get Window Size """
        val = 0
        for item in items:
            val += item
        return val

    @staticmethod
    def getHanning(numSamples,headPad=None,tailPad=None):
        """ Get a Hanning Window of the Specified Size """
        if (headPad is None):
            headPad = WindowFunctions.PadHead
        if (tailPad is None):
            tailPad = WindowFunctions.PadTail
        window = np.zeros(
            shape=(WindowFunctions.windowSize(numSamples,headPad,tailPad),),
            dtype=np.float32)
        window[headPad:tailPad] = scisig.windows.hann(numSamples)
        return window

