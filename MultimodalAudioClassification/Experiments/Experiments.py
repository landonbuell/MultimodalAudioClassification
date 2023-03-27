"""
    Repo:       MultimodalAudioClassification
    Solution:   MultimodalAudioClassification
    Project:    Experiments
    File:       ExperimentDrivers.py

    Author:     Landon Buell
    Date:       November 2022
"""

    #### IMPORTS ####

import sys
import os

import numpy as np

import ExperimentCallbacks
import ModelParams

import PyToolsStructures
import Preprocessors


    #### CONSTANTS ####


    #### CLASS DEFINITIONS ####

class __BaseExperiment:
    """ Parent Class for Running Experiments """

    def __init__(self,
                 runInfo,
                 outputPath,
                 modelLoaderCallback,       # <model> = modelLoaderCallback.__call__(self,randomSeed)
                 preprocessCallbacks=None,
                 pipelines=[],
                 trainSize=0.8,
                 numIters=8,    
                 epochsPerBatch=2,
                 seed=123456789):
        """ Constructor """
        self._runInfo       = runInfo
        self._outputPath    = os.path.abspath(outputPath)

        if (os.path.isdir(self._outputPath) == False):
            os.makedirs(self._outputPath)
        
        self._modelLoaderCallback       = modelLoaderCallback
        self._preprocessCallbacks       = []
        if (preprocessCallbacks is not None):
            self._preprocessCallbacks = preprocessCallbacks[:]

        self._numIters      = numIters
        self._seed          = seed

        self._pipelines     = pipelines
        self._trainSize     = trainSize

        self._model     = None
        self._scaler    = Preprocessors.StandardScalerWrapper(runInfo)
        for pipelineIndex in self._pipelines:
            self._scaler.loadParams(pipelineIndex)

        self._fitParams = ModelParams.TensorFlowFitModelParams()
        self._fitParams.callbacks.append(ExperimentCallbacks.TrainingLoggerCallback(self))
        self._fitParams.epochs = epochsPerBatch
        self._predictParams = ModelParams.TensorFlowPredictModelParams()
        self._predictParams.callbacks.append(ExperimentCallbacks.TestingLoggerCallback(self))

        self._trainingBatches   = np.array([],dtype=np.int32)
        self._testingBatches    = np.array([],dtype=np.int32)

        self._trainingHistories = []
        self._trainingMetrics   = ModelParams.ModelTrainingMetrics()
        self._testingMetrics    = ModelParams.ModelTestingMetrics(runInfo.getNumClasses())
        
        self._extraData = dict()    # <str> -> <any>

    def __del__(self):
        """ Destructor """
        self._runInfo   = None
        self._model     = None

    # Getters and Setters

    def getRunInfo(self):
        """ Return the RunInfo Structure """
        return self._runInfo

    def getPipelines(self):
        """ Return a list of the pipelines to load """
        return self._pipelines

    def updateTrainingMetricsWithLog(self,batchLog):
        """ Use a batch log to update training metric data """
        self._trainingMetrics.updateWithBatchLog(batchLog)
        return None

    def updateTestingPredictionsWithLog(self,batchLog):
        """ Use a batch log to update prediction data """
        self._testingMetrics.updateWithBatchLog(batchLog)
        return None

    def getExtraData(self,extraDataKey: str):
        """ Fetch any additional data stored in this instance by key """
        keys = self._extraData.keys()
        if (extraDataKey not in keys):
            return None
        return self._extraData[extraDataKey]

    # Public Interface

    def registerTrainingBatches(self,batches):
        """ Add a list of batches to the training batch list """
        self._trainingBatches = np.append(self._trainingBatches,batches)
        return self

    def registerTestingBatches(self,batches):
        """ Add a list of batches to the testing batch list """
        self._testingBatches = np.append(self._testingBatches,batches)
        return self

    def run(self):
        """ Run the Experiment """

        # Initialize Model + Train Test Split
        self.__initializeModel()
        self.__registerTrainTestBatches()

        # Train the Model + Export the Data
        for ii in range(self._numIters):          
            self.__runLoadAndTrainSequence()
        self.__exportTrainingDetails()

        # Test the Model + Export the Data
        self.__runLoadAndTestSequence()
        self.__exportTestingDetails()

        return self

    def resetState(self):
        """ Public accessor to reset the state of the experiment instance """
        return self.__resetState()

    def predictWithModel(self,X):
        """ Run predictions on Model """
        return self

    # Protected Interface

    # Private Interface
    
    def __initializeModel(self):
        """ Initialize the Neural Network Model """
        randomState = self._seed
        self._model = self._modelLoaderCallback.__call__(self)
        if (self._model is None):
            errMsg = "Got NoneType for model instance"
            raise RuntimeError(errMsg)
        return self

    def __loadTrainBatch(self,batchIndex):
        """ Load + Return a Batch of Data """
        pipelinesToLoad = self.getPipelines()
        numClasses = self.getRunInfo().getNumClasses()
        designMatrices = self.getRunInfo().loadSingleBatchFromPipelines(
            batchIndex,pipelinesToLoad)
        X = [designMatrices[ii].getFeatures() for ii in pipelinesToLoad]
        Y = [designMatrices[ii].getLabels() for ii in pipelinesToLoad]
        Y = [ExperimentCallbacks.oneHotEncode(y,numClasses) for y in Y]
        return (X,Y)

    def __loadTestBatch(self,batchIndex):
        """ Load + Return a Batch of Data """
        pipelinesToLoad = self.getPipelines()
        numClasses = self.getRunInfo().getNumClasses()
        designMatrices = self.getRunInfo().loadSingleBatchFromPipelines(
            batchIndex,pipelinesToLoad)
        X = [designMatrices[ii].getFeatures() for ii in pipelinesToLoad]
        Y = [designMatrices[ii].getLabels() for ii in pipelinesToLoad]
        return (X,Y)

    def __registerTrainTestBatches(self):
        """ Determine which batches will be used for training/testing """
        totalNumBatches = self._runInfo.getNumBatches()
        numTrainBatches = int(totalNumBatches * self._trainSize)
        batches = np.arange(totalNumBatches)
        np.random.shuffle(batches)
        self._trainingBatches = batches[0:numTrainBatches]
        self._testingBatches = batches[numTrainBatches:]
        return self

    def __applyStandardScaling(self,designMatrices: list):
        """ Apply a Standard Scaler to Inputs designMatrices """
        if (len(designMatrices) != len(self._pipelines)):
            msg = "Inconsistent number of pipleines + provided design matrices."
            raise RuntimeError(msg)
        # Apply scaler
        designMatricesScaled = []
        for (matrix,pipelineIndex) in zip(designMatrices,self._pipelines):
            X = self._scaler.applyFitToMatrix(matrix,pipelineIndex)
            designMatricesScaled.append(X)
        return designMatricesScaled

    def __executePreprocessingCallbacks(self,X: np.ndarray,Y: np.ndarray):
        """ Execute the list of preprocessing callbacks """
        for callback in self._preprocessCallbacks:
            X,Y = callback.__call__(X,Y)
        return X,Y

    def __runLoadAndTrainSequence(self):
        """ Run data loading/training sequence """
        for batchIndex in self._trainingBatches:
            X,Y = self.__loadTrainBatch(batchIndex)
            X = self.__applyStandardScaling(X)
            X,Y = self.__executePreprocessingCallbacks(X,Y)

            # Set any fit params
            self._fitParams.batchSize = X[0].shape[0]

            # Fit the Model
            trainingHistory = self._model.fit(
                x=X,
                y=Y,
                batch_size=self._fitParams.batchSize,
                epochs=self._fitParams.epochs,
                verbose=self._fitParams.verbose,
                callbacks=self._fitParams.callbacks,
                shuffle=self._fitParams.shuffle)
            self._trainingHistories.append(trainingHistory)

        # Done 
        return self

    def __runLoadAndTestSequence(self):
        """ Run data loading/testing sequence """
        for batchIndex in self._testingBatches:
            X,Y_truth = self.__loadTestBatch(batchIndex)
            X = self.__applyStandardScaling(X)
            X,Y_truth = self.__executePreprocessingCallbacks(X,Y_truth)

            # Set any predict Params
            self._predictParams.batchSize = X[0].shape[0]
            
            # predict
            Y_preds = self._model.predict(
                x=X,
                batch_size=self._predictParams.batchSize,
                callbacks=self._predictParams.callbacks)
            self._testingMetrics.updateWithPredictionData(Y_truth,Y_preds)

        # Done
        return self

    def __exportTrainingDetails(self):
        """ Export the Details of the Training Process """
        frame = self._trainingMetrics.toDataFrame()
        exportPath = os.path.join(self._outputPath,"trainingHistory.csv")
        frame.to_csv(exportPath)
        return self

    def __exportTestingDetails(self):
        """ Export the Details of the Testing Process """
        frame = self._testingMetrics.toDataFrame()
        exportPath = os.path.join(self._outputPath,"testResults.csv")
        frame.to_csv(exportPath,index=None)
        return self

    def __resetState(self):
        """ Reset the State of the experiment in between iterations """
        self._seed * (2.0/3.0)
        self._model = None
        self._trainingBatches   = np.array([],dtype=np.int32)
        self._testingBatches    = np.array([],dtype=np.int32)
        self._trainingHistories.clear()
        return self

class MultilayerPerceptronExperiment(__BaseExperiment):
    """ Train + Test Multilater perceptron """
    
    def __init__(self,
                 runInfo,
                 outputPath,
                 trainSize=0.8,
                 numIters=1,              
                 seed=123456789):
        """ Constructor """
        super().__init__(runInfo,
                         outputPath,
                         modelLoaderCallback=ExperimentCallbacks.ModelLoaderCallbacks.loadMultilayerPerceptron,
                         pipelines=[0],
                         trainSize=trainSize,
                         numIters=numIters,
                         seed=seed)

    def __del__(self):
        """ Destructor """
        super().__del__()

class ConvolutionalNeuralNetworkExperiment(__BaseExperiment):
    """ Train + Test Multilater perceptron """
    
    def __init__(self,
                 runInfo,
                 outputPath,
                 trainSize=0.8,
                 numIters=1,              
                 seed=123456789):
        """ Constructor """
        super().__init__(runInfo,
                         outputPath,
                         modelLoaderCallback=ExperimentCallbacks.ModelLoaderCallbacks.loadConvolutionalNeuralNetwork,
                         pipelines=[1],
                         trainSize=trainSize,
                         numIters=numIters,
                         seed=seed)
        self._preprocessCallbacks.append(ExperimentCallbacks.DataPreprocessingCallbacks.reshapePipeline2Features)

    def __del__(self):
        """ Destructor """
        super().__del__()



