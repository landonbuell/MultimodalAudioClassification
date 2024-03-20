"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionApp
    File:       collectionSession.py
    Classes:    CollectionSession

    Author:     Landon Buell
    Date:       March 2024
"""


        #### IMPORTS ####

import os
import threading

import featureCollectionApp

        #### CLASS DEFINITIONS ####

class FeatureCollectionSession:
    """ Encapsulates a collection session """

    def __init__(self,
                 numThreads: int):
        """ Constructor """
        self._collectors = [None] * numThreads

    def __del__(self):
        """ Destructor """
        for thread in self._threads:
            if 