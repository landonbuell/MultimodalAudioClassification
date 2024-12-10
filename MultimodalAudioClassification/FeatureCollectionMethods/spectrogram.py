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
                 separateRealAndImag=True):
        """ Constructor """
        super().__init__(Spectrogram.__NAME,
                            frameParams.getFreqFramesNumFeatures(
                            separateRealImag=separateRealAndImag)) # set to 1 for memory efficiency!
        self._params = frameParams
        self._separateRealAndImaginary = separateRealAndImag

        intendedShape = list(
            self._params.getFreqFrameShape(
                self._separateRealAndImaginary))
        self._setIntendedShape(intendedShape)
        
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
        return self._params.getFreqFrameSizeUnmasked()

    @property
    def numChannels(self) -> int:
        """ Return the number of channels in the """
        return int(self._separateRealAndImaginary) + 1

    # Public Interface

    def featureNames(self) -> list:
        """ VIRTUAL: Return a list of the feature names """
        result = [None] * self.getNumFeatures()    
        counter = 0
        outputShape = self._params.getFreqFrameShape(self._separateRealAndImaginary)
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
                  signal: collectionMethod.signalData.SignalData,
                  features: collectionMethod.featureVector.FeatureVector) -> bool:
        """ OVERRIDE: main body of call function """
        signal.makeFreqSeriesAnalysisFrames(self._params)
        #signal.cachedData.analysisFramesFreq.plot(signal.getSourcePath())
        if (self._separateRealAndImaginary == True):
            features.appendItems( np.real(signal.cachedData.analysisFramesFreq.rawFrames().ravel()) )
            features.appendItems( np.imag(signal.cachedData.analysisFramesFreq.rawFrames().ravel()) )
        else:
            features.appendItems( np.abs(signal.cachedData.analysisFramesFreq.rawFrames()) )
        return True
    