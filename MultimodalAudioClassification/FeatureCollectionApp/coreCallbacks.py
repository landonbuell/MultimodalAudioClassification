"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       featureVector.py
    Classes:    SignalDataPreprocessCallbacks,
                FeatureVectorPreprocessCallbacks,
                SignalDataPostprocessCallbacks,
                FeatureVectorPostprocessCallbacks

    Author:     Landon Buell
    Date:       February 2024
"""


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
    pass