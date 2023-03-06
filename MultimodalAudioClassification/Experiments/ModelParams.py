"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    Experiments
File:       ModelParams.py

Author:     Landon Buell
Date:       November 2022
"""

        #### IMPORTS ####

        #### CLASS DEFINITIONS ####

class TensorFlowFitModelParams:
    """ Structure to store Params for fitting a TF Model """

    def __init__(self):
        """ Construction """
        self.batchSize  = None
        self.epochs     = 1
        self.verbose    = 'auto'
        self.callbacks  = None
        self.splitValidation = 0.0
        self.dataValidation = None
        self.shuffle    = True
        self.classWeight = None
        self.initialEpoch = 0

    def __del__(self):
        """ Destructor """
        pass

