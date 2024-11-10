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
        pipeline = featurePipeline.FeaturePipeline("pipeline00All")
        pipeline.appendCollectionMethod( timeDomainEnvelope.TimeDomainEnvelope(16) )
        pipeline.appendCollectionMethod( zeroCrossingRate.TotalZeroCrossingRate() )
        pipeline.appendCollectionMethod( zeroCrossingRate.FrameZeroCrossingRate() )
        pipeline.appendCollectionMethod( centerOfMass.TemporalCenterOfMass(
            centerOfMass.collectionMethod.WeightingFunction.LINEAR ) )
        pipeline.appendCollectionMethod( centerOfMass.TemporalCenterOfMass(
            centerOfMass.collectionMethod.WeightingFunction.LOG_NATURAL) )
        pipeline.appendCollectionMethod( centerOfMass.TemporalCenterOfMass(
            centerOfMass.collectionMethod.WeightingFunction.LINEAR) )
        pipeline.appendCollectionMethod( centerOfMass.TemporalCenterOfMass(
            centerOfMass.collectionMethod.WeightingFunction.LOG_NATURAL) )
        pipeline.appendCollectionMethod( autoCorrelation.AutoCorrelationCoefficients(16) )
        pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergyMeans( params, 32 ) )
        pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficientMeans( params, 32 ) )
        return pipeline

    @staticmethod
    def getDefaultPipeline01() -> featurePipeline.FeaturePipeline:
        """ Get the default pipeline 01 """
        params = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        pipeline = featurePipeline.FeaturePipeline("pipeline01Mfbes")
        pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergies( params, 16 ) )
        pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergyMeans( params, 16 ) )
        pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergyVaris( params, 16 ) )
        pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergyMedians( params, 16 ) )
        pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergyMinMax( params, 16 ) )
        return pipeline

    @staticmethod
    def getDefaultPipeline02() -> featurePipeline.FeaturePipeline:
        """ Get the default pipeline 02 """
        params = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        pipeline = featurePipeline.FeaturePipeline("pipeline02Mfccs")
        pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficients( params, 16 ) )
        pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficientMeans( params, 16 ) )
        pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficientVaris( params, 16 ) )
        pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficientMedians( params, 16 ) )
        pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficientMinMax( params, 16 ) )
        return pipeline

    @staticmethod
    def getDefaultPipeline03() -> featurePipeline.FeaturePipeline:
        """ Get the default pipeline 03 """
        frameParams = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        pipeline = featurePipeline.FeaturePipeline("pipeline03Spectogram")
        pipeline.appendCollectionMethod( spectrogram.Spectrogram(frameParams) )
        return pipeline