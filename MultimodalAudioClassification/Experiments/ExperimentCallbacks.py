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

import NeuralNetworks

        #### FUNCTION DEFINITIONS ####

def oneHotEncode(targets,numClasses):
    """ One-hot-encode a vector of targets """
    Y = np.zeros(shape=(len(targets),numClasses),dtype=np.int16)
    for ii,tgt in enumerate(targets):
        Y[ii,tgt] = 1
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
        numClasses  = np.max(runInfo.getClassesInUse())
        model = NeuralNetworks.NeuralNetworkPresets.getDefaultModelMultilayerPerceptron(
            inputShape,numClasses,"MLP")
        return model


class DataLoaderCallbacks:
    """ 
        Static class of Data Loader Callbacks 
        Signatures
            X,Y = func(experiment,batchIndex)
            
            
    """

    def loadPipelineBatch(experiment,batchIndex):
        """ Load a Batch from a particilar pipeline """
        pipelinesToLoad = experiment.getPipelines()
        numClasses = np.max(experiment.getRunInfo().getClassesInUse())
        designMatrices = experiment.getRunInfo().loadSingleBatchFromPipelines(
            batchIndex,pipelinesToLoad)
        X = [designMatrices[ii].getFeatures() for ii in pipelinesToLoad]
        Y = [designMatrices[ii].getLabels() for ii in pipelinesToLoad]
        Y = [oneHotEncode(y,numClasses) for y in Y]
        return (X,Y)
