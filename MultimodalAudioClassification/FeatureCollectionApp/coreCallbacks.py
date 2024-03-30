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

        #### IMPORTS ####

import featurePipeline

import timeDomainEnvelope
import zeroCrossingRate
import centerOfMass
import autoCorrelation

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

class DefaultFeaturePipeline:
    """ Static Class of Default Feature Pipelines """

    def getDefaultPipeline00() -> featurePipeline.FeaturePipeline:
        """ Get the default pipeline 00 """
        pipeline = featurePipeline.FeaturePipeline("Alpha")
        pipeline.appendCollectionMethod( timeDomainEnvelope.TimeDomainEnvelope(12) )
        pipeline.appendCollectionMethod( zeroCrossingRate.TotalZeroCrossingRate() )
        pipeline.appendCollectionMethod( centerOfMass.TemporalCenterOfMass(
                                            centerOfMass.collectionMethod.WeightingFunction.LINEAR) )
        pipeline.appendCollectionMethod( autoCorrelation.AutoCorrelationCoefficients(16) )

        return pipeline

    def getDefaultPipeline01() -> featurePipeline.FeaturePipeline:
        """ Get the default pipeline 01 """
        pipeline = featurePipeline.FeaturePipeline("Beta")
        return pipeline