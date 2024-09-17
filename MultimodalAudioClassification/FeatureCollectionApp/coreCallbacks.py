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
import analysisFrames

import timeDomainEnvelope
import zeroCrossingRate
import centerOfMass
import autoCorrelation
import melFilterBankEnergies
import cepstralCoefficients
import spectrogram

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

    @staticmethod
    def getDefaultPipeline00() -> featurePipeline.FeaturePipeline:
        """ Get the default pipeline 00 """
        params = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        pipeline = featurePipeline.FeaturePipeline("Alpha")
        #pipeline.appendCollectionMethod( timeDomainEnvelope.TimeDomainEnvelope(12) )
        #pipeline.appendCollectionMethod( zeroCrossingRate.TotalZeroCrossingRate() )
        #pipeline.appendCollectionMethod( centerOfMass.TemporalCenterOfMass(
        #                                    centerOfMass.collectionMethod.WeightingFunction.LINEAR) )
        #pipeline.appendCollectionMethod( autoCorrelation.AutoCorrelationCoefficients(16,16) )
        pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergyMeans(
            params, 12 ) )
        pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficients(
            params, 12 ) )
        return pipeline

    @staticmethod
    def getDefaultPipeline01() -> featurePipeline.FeaturePipeline:
        """ Get the default pipeline 01 """
        frameParams = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        pipeline = featurePipeline.FeaturePipeline("Beta")
        pipeline.appendCollectionMethod( spectrogram.Spectrogram(frameParams) )
        return pipeline