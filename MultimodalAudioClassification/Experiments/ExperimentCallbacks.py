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

        #### FUNCTION DEFINITIONS ####

def oneHotEncode(targets,numClasses):
    """ One-hot-encode a vector of targets """
    Y = np.zeros(shape=(len(targets),numClasses),dtype=np.int16)
    for ii,tgt in targets:
        Y[ii,tgt] = 1
    return Y

        #### CLASS DEFINITIONS ####

class ModelLoaderCallbacks:
    """ Static class of Model Loader Callbacks """

    @staticmethod
    def loadMultilayerPerceptron():
        """ Load in Multilayer Perceptron """
        return None


class DataLoaderCallbacks:
    """ 
        Static class of Data Loader Callbacks 
        Signatures
            X,Y = func(experiment,batchIndex)
            
            
    """

    def loadPipelineBatch(experiment,batchIndex):
        """ Load a Batch from a particilar pipeline """
        pipelinesToLoad = experiment.getPipelinesToLoad()
        numClasses = np.max(experiment.getRunInfo().getClassesInUse())
        designMatrices = experiment.getRunInfo().loadSingleBatchFromPipelines(
            batchIndex,pipelinesToLoad)
        X = [designMatrices[ii].getFeatures() for ii in pipelinesToLoad]
        Y = [designMatrices[ii].getLabels() for ii in pipelinesToLoad]
        Y = [oneHotEncode(y,numClasses) for y in Y]
        return (X,Y)
