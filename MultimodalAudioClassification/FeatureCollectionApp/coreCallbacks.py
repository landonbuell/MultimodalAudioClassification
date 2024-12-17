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

class DefaultFeaturePipeline:
    """ Static Class of Default Feature Pipelines """

    @staticmethod
    def getAllFeatures() -> featurePipeline.FeaturePipeline:
        """Return a pipeline that returns 'all' 1D features """
        params = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        pipeline = featurePipeline.FeaturePipeline("allFeatures")
        pipeline.appendCollectionMethod( timeDomainEnvelope.TimeDomainEnvelope(16) )
        pipeline.appendCollectionMethod( zeroCrossingRate.TotalZeroCrossingRate() )
        pipeline.appendCollectionMethod( zeroCrossingRate.FrameZeroCrossingRate(params) )
        pipeline.appendCollectionMethod( centerOfMass.TemporalCenterOfMass(
            centerOfMass.collectionMethod.WeightingFunction.LINEAR ) )
        pipeline.appendCollectionMethod( centerOfMass.TemporalCenterOfMass(
            centerOfMass.collectionMethod.WeightingFunction.QUADRATIC) )
        pipeline.appendCollectionMethod( centerOfMass.TemporalCenterOfMass(
            centerOfMass.collectionMethod.WeightingFunction.LOG_NATURAL) )
        pipeline.appendCollectionMethod( centerOfMass.TemporalCenterOfMass(
            centerOfMass.collectionMethod.WeightingFunction.LOG_BASE10) )
        pipeline.appendCollectionMethod( autoCorrelation.AutoCorrelationCoefficients(16) )
        #pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergyMeans( params, 32 ) )
        #pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficientMeans( params, 32 ) )
        return pipeline

    @staticmethod
    def getMelFilterBankEnergyInfo() -> featurePipeline.FeaturePipeline:
        """ Get the pipeline that returns Info on MFBE's """
        params = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        pipeline = featurePipeline.FeaturePipeline("melFilterBankEnergyInfo")
        #pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergies(params, 16) )
        pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergyMeans( params, 16 ) )
        pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergyVaris( params, 16 ) )
        pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergyMedians( params, 16 ) )
        pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergyMinMax( params, 16 ) )
        return pipeline

    @staticmethod
    def getMelFilterBankEnergyMatrix() -> featurePipeline.FeaturePipeline:
        """ Get the pipeline that returns the MFBE matrix """
        params = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        pipeline = featurePipeline.FeaturePipeline("melFilterBankEnergyMatrix")
        pipeline.appendCollectionMethod( melFilterBankEnergies.MelFilterBankEnergies(params, 16) )
        return pipeline

    @staticmethod
    def getMelFrequencyCepstralCoefficientsInfo() -> featurePipeline.FeaturePipeline:
        """ Get the pipeline that returns Info on MFCC's """
        params = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        pipeline = featurePipeline.FeaturePipeline("melFrequencyCepstralCoefficientsInfo")
        #pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficients( params, 16 ) )
        pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficientMeans( params, 16 ) )
        pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficientVaris( params, 16 ) )
        pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficientMedians( params, 16 ) )
        pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficientMinMax( params, 16 ) )
        return pipeline

    @staticmethod
    def getMelFrequencyCepstralCoefficientsMatrix() -> featurePipeline.FeaturePipeline:
        """ Get the pipeline that returns the MFCC matrix """
        params = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        pipeline = featurePipeline.FeaturePipeline("melFrequencyCepstralCoefficientsMatrix")
        pipeline.appendCollectionMethod( cepstralCoefficients.MelFrequencyCepstrumCoefficients( params, 16 ) )
        return pipeline

    @staticmethod
    def getSpectrogram() -> featurePipeline.FeaturePipeline:
        """ Get the pipeline that returns the spectrogram matrix """
        frameParams = analysisFrames.AnalysisFrameParameters.defaultFrameParams()
        pipeline = featurePipeline.FeaturePipeline("spectrogram")
        pipeline.appendCollectionMethod( spectrogram.Spectrogram(frameParams) )
        return pipeline