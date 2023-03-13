"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    Experiments
File:       ModelParams.py

Author:     Landon Buell
Date:       November 2022
"""

        #### IMPORTS ####

import numpy as np
import pandas as pd

        #### CLASS DEFINITIONS ####

class TensorFlowFitModelParams:
    """ Structure to store Params for fitting a TF Model """

    def __init__(self):
        """ Constructor """
        self.batchSize  = None
        self.epochs     = 1
        self.verbose    = 'auto'
        self.callbacks  = []
        self.splitValidation = 0.0
        self.dataValidation = None
        self.shuffle    = True
        self.classWeight = None
        self.initialEpoch = 0

    def __del__(self):
        """ Destructor """
        pass

class TensorFlowPredictModelParams:
    """ Structure to store Params for prediciting a TF model """
    
    def __init__(self):
        """ Constructor """
        self.batchSize  = None
        self.verbose    = 'auto'
        self.steps      = None
        self.callbacks  = []
        
    def __del__(self):
        """ Destructor """
        pass


class ModelTrainingMetrics:
    """ Stores arrays of all of the metrics """

    def __init__(self):
        """ Constructor """
        self._iterCount  = 0
        self._loss       = np.array([],dtype=np.float64)
        self._accuracy   = np.array([],dtype=np.float64)
        self._precision  = np.array([],dtype=np.float64)
        self._recall     = np.array([],dtype=np.float64)

    def __del__(self):
        """ Destructor """
        self._iterCount  = None
        self._loss       = None
        self._accuracy   = None
        self._precision  = None
        self._recall     = None

    def updateWithBatchLog(self,batchLog):
        """ Update instance w/ data from batch Log """
        self._iterCount  += 1
        self._loss       = np.append(self._loss,batchLog['loss'])
        self._accuracy   = np.append(self._accuracy,batchLog['accuracy'])
        self._precision  = np.append(self._precision,batchLog['precision'])
        self._recall     = np.append(self._recall,batchLog['recall'])
        return self

    def toDataFrame(self) -> pd.DataFrame:
        """ Return this Structure as a PD DataFrame """
        indexCol = np.arange(self._iterCount)
        dataMap = {"Loss"       : self._loss,
                   "Accuracy"   : self._accuracy,
                   "Precision"  : self._precision,
                   "Recall"     : self._recall}
        frame = pd.DataFrame(
            data=dataMap,
            index=indexCol)
        return frame

class ModelTestingMetrics:
    """ Stores Array of Testing Predictions """

    def __init__(self,numClasses):
        """ Constructor """
        self._numClasses = numClasses
        self._labels    = np.empty(shape=(0,),dtype=np.uint16)
        self._preds     = np.empty(shape=(0,numClasses),dtype=np.float32)

    def __del__(self):
        """ Destructor """
        self._labels = None
        self._preds = None
    
    def updateWithBatchLog(self,batchLog):
        """ Update instacne w/ data from a prediction log """
        return None

    def updateWithPredictionData(self,labels,predictions):
        """ Update instance w/ data from labels + predictions """
        predictionsShapeOld = self._preds.shape
        predictionsShapeNew = predictions.shape
        numSamplesTotal = predictionsShapeOld[0] + predictionsShapeNew[0]
        predictionsShapeNow = (numSamplesTotal,predictionsShapeOld[1])

        self._labels = np.append(self._labels,labels)
        self._preds = np.append(self._preds,predictions)
        self._preds = np.reshape(self._preds,newshape=predictionsShapeNow)
        return None



    def toDataFrame(self) -> pd.DataFrame:
        """ Return this structure as a PD DataFrame """
        return pd.DataFrame()
