"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    ExperimentCallbacks
File:       main.py

Author:     Landon Buell
Date:       January 2023
"""

        #### IMPORTS ####

import numpy as np
import tensorflow as tf

import NeuralNetworks

        #### FUNCTION DEFINITIONS ####

def oneHotEncode(targets,numClasses):
    """ One-hot-encode a vector of targets """
    Y = np.zeros(shape=(len(targets),numClasses),dtype=np.int16)
    for ii,tgt in enumerate(targets):
        Y[ii,tgt] = 1
    return Y

def reverseOneHotEncode(targets):
    """ Reverse one-hot-encode a matrix """
    numSamples = targets.shape[0]
    Y = np.empty(shape=numSamples,dtype=np.int16)
    for ii,row in enumerate(targets):
        tgt = np.argmax(row)
        Y[ii] = tgt
    return Y
    


        #### CLASS DEFINITIONS ####

class ModelLoaderCallbacks:
    """ Static class of Model Loader Callbacks """

    @staticmethod
    def loadMultilayerPerceptron(experiment):
        """ Load in Multilayer Perceptron """
        pipelineIndex = 0
        runInfo     = experiment.getRunInfo()
        inputShape  = runInfo.getSampleShapeOfPipeline(pipelineIndex)
        numClasses  = runInfo.getNumClasses()
        model = NeuralNetworks.NeuralNetworkPresets.getDefaultModelMultilayerPerceptron(
            inputShape,numClasses,"MLP")
        return model

    @staticmethod
    def loadConvolutionalNeuralNetwork(experiment):
        """ Load in Convolutional Neural Network """
        pipelineIndex = 1
        runInfo     = experiment.getRunInfo()
        inputShape  = runInfo.getSampleShapeOfPipeline(pipelineIndex)
        numClasses  = runInfo.getNumClasses()
        model = None
        return model

class DataLoaderCallbacks:
    """ 
        Static class of Data Loader Callbacks 
        Signatures
            X,Y = func(experiment,batchIndex)        
    """

    @staticmethod
    def loadPipelineBatchForTraining(experiment,batchIndex):
        """ Load a Batch from a particular pipeline + One Hot Encode"""
        pipelinesToLoad = experiment.getPipelines()
        numClasses = experiment.getRunInfo().getNumClasses()
        designMatrices = experiment.getRunInfo().loadSingleBatchFromPipelines(
            batchIndex,pipelinesToLoad)
        X = [designMatrices[ii].getFeatures() for ii in pipelinesToLoad]
        Y = [designMatrices[ii].getLabels() for ii in pipelinesToLoad]
        Y = [oneHotEncode(y,numClasses) for y in Y]
        return (X,Y)

    @staticmethod
    def loadPipelineBatchForTesting(experiment,batchIndex):
        """ Load a Batch from a particular pipeline + Do-not One Hot Encode"""
        pipelinesToLoad = experiment.getPipelines()
        numClasses = experiment.getRunInfo().getNumClasses()
        designMatrices = experiment.getRunInfo().loadSingleBatchFromPipelines(
            batchIndex,pipelinesToLoad)
        X = [designMatrices[ii].getFeatures() for ii in pipelinesToLoad]
        Y = [designMatrices[ii].getLabels() for ii in pipelinesToLoad]
        return (X,Y)

class TrainingLoggerCallback(tf.keras.callbacks.Callback):
    """ Logs training data to be saved """

    def __init__(self,experiment):
        """ Constructor """
        super().__init__()
        self._experiment = experiment

    def __del__(self):
        """ Destructor """
        pass

    # Callbacks

    def on_train_batch_end(self,batchIndex,logs=None):
        """ Behavior for the end of each batch """
        self._experiment.updateTrainingMetricsWithLog(logs)
        return None

class TestingLoggerCallback(tf.keras.callbacks.Callback):
    """ Logs prediction data to be saved """

    def __init__(self,experiment):
        """ Constructor """
        super().__init__()
        self._experiment = experiment

    def __del__(self):
        """ Destructor """
        pass

    def on_predict_batch_end(self,batchIndex,logs=None):
        """ Behavior for the end of each batch """
        self._experiment.updateTestingPredictionsWithLog(logs)
        return None