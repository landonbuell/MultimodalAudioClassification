"""
Repository:     MultimodalAudioClassification
Solution:       MultimodalAudioClassification
Project:        FeatureCollection  
File:           Administrative.py
 
Author:         Landon Buell
Date:           January 2023
"""

        #### IMPORTS ####

import PyToolsPlotting

        #### CLASS DEFINTIONS ####

class SignalDataPreprocessCallbacks:
    """ Static Class - Make No Instance """

    @staticmethod
    def makeAnalysisFramesTime(pipeline,signalData):
        """ Use Frame Params to Make signalData Analysis Frames """
        frameParams = pipeline.getAnalysisFrameParams()
        signalData.makeAnalysisFramesTime(frameParams)
        return None

    @staticmethod
    def makeAnalysisFramesFreq(pipeline,signalData):
        """ Use Frame Params to Make signalData Analysis Frames """
        frameParams = pipeline.getAnalysisFrameParams()
        signalData.makeAnalysisFramesFreq(frameParams)
        return None

class FeatureVectorPostProcessCallbacks:
    """ Static Class - Make no Instance """

    def plotSpectrogram(pipeline,featureVector):
        """ Plot the spectrogram from the feature vector """
        frameParams = pipeline.getAnalysisFrameParams()
        freqFramesShape = frameParams.getFreqFramesShape()

        # Assemble Data For Analysis Frames
        X = featureVector.getData().reshape(freqFramesShape)
        timeAxis = frameParams.generateTimeAxis()
        freqAxis = frameParams.generateFreqAxis()
        PyToolsPlotting.spectrogram(
            X,timeAxis,freqAxis,"Spectrogram")
        return None