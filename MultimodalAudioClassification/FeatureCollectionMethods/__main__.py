"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollectionMethods
    File:       __main__.py
    Classes:    -

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import os
import sys

import unitTests

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":
    
    # Create the unit Tests
    testSuite = unitTests.PresetUnitTests.getTestBasicTimeSeriesMethods()
    testSuite.runAll()

    pass # temp
