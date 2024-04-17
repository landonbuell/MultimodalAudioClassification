"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       spectrogram.py
    Classes:    Spectrogram,

    Author:     Landon Buell
    Date:       February 2024
"""


        #### IMPORTS ####

import numpy as np

import signalData

import collectionMethod
import analysisFrames

        #### CLASS DEFINITIONS ####

class Spectrogram(collectionMethod.AbstractCollectionMethod):
    """ 
        Compute and Store the spectrogram of a signal
    """

    __NAME = "SPECTROGRAM"
    
    def __init__(self,
                 frameParams: analysisFrames.AnalysisFrameParameters,
                 separateRealImag=True):
        """ Constructor """
        super().__init__(Spectrogram.__NAME,frameParams.getFreqFramesNumFeatures(separateRealImag=separateRealImag)) # set to 1 for memory efficiency!
        self._params = frameParams
        self._separateRealImage = separateRealImag
        
    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    @property
    def numFrames(self) -> int:
        """ Return the max allowed number of frames """
        return self._params.maxNumFrames

    @property
    def frameSize(self) -> int:
        """ Return the size of each frame """
        return self._params.freqFrameSize

    # Public Interface

    def featureNames(self) -> list:
        """ VIRTUAL: Return a list of the feature names """
        result = [None] * self.getNumFeatures()    
        counter = 0
        outputShape = self._params.getFreqFrameShape(self._separateRealImage)
        # output Shape is always 3D
        for ii in range(outputShape[0]):
            for jj in range(outputShape[1]):
                for kk in range(outputShape[2]):
                    name = "[{0},{1},{2}]".format(ii,jj,kk)
                    result[counter] = name
                    counter += 1
        return result

    # Protected Interface

    def _callBody(self, 
                  signal: signalData.SignalData) -> bool:
        """ OVERRIDE: main body of call function """
        signal.makeFreqSeriesAnalysisFrames(self._params)
        #signal.cachedData.analysisFramesFreq.plot(signal.getSourcePath())
        if (self._separateRealImage == True):
            halfWay = int(self.getNumFeatures() / 2)
            np.copyto(
                self._data[:halfWay],
                np.real(signal.cachedData.analysisFramesFreq.rawFrames().ravel()),
                casting='no')
            np.copyto(
                self._data[halfWay:],
                np.imag(signal.cachedData.analysisFramesFreq.rawFrames().ravel()),
                casting='no')
        else:
             np.copyto(
                 self._data,
                 np.abs(signal.cachedData.analysisFramesFreq.rawFrames()),
                 casting='no')
        return True
    