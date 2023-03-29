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
        #inputShape  = runInfo.getSampleShapeOfPipeline(pipelineIndex)
        inputShape  = (256,1115,1)
        numClasses  = runInfo.getNumClasses()
        model = NeuralNetworks.NeuralNetworkPresets.getDefaultModelConvolutionalNeuralNetwork(
            inputShape,numClasses,"CNN")
        return model

    @staticmethod
    def loadHybridNeuralNetwork(experiment):
        """ Load in Hybrid NeuralNetwork """
        runInfo = experiment.getRunInfo()
        inputShapeA = runInfo.getSampleShapeOfPipeline(0)
        inputShapeB = (256,1115,1)
        numClasses = runInfo.getNumClasses()
        model = NeuralNetworks.NeuralNetworkPresets.getDefaultHybridModel(
            inputShapeA,inputShapeB,numClasses,"HNN")
        return model
            

class DataPreprocessingCallbacks:
    """ Static class with methods used to preprocess data """

    @staticmethod
    def reshapePipeline2Features(X,Y):
        """ Reshape the pipeline #2's Features """
        if (len(X) == 1):
            batchSize = X[0].shape[0]
            newShape = [batchSize] + [256,1115,1]
            X[0] = np.reshape(X[0],newshape=newShape)
        elif (len(X) >= 2):
            batchSize = X[1].shape[0]
            newShape = [batchSize] + [256,1115,1]
            X[1] = np.reshape(X[1],newshape=(256,1115,1))
        else:
            errMsg = "Got empty input Array"
            raise RuntimeError(errMsg)
        return X,Y

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