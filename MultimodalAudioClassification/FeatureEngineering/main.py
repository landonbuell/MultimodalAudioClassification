"""
Repo:       MultiModalAudioClassification
Solution:   MultiModalAudioClassification
Project:    FeautureEngineering
File:       main.py

Author:     Landon Buell
Date:       Sept 2022
"""

    #### IMPORTS ####

import sys

import PyToolsStructures

    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set some constants + Load Run Info
    RUN_PATH = "C:\\Users\\lando\\Documents\\audioFeatures\\simpleSignalsV1"
    runInfo = PyToolsStructures.RunInfo.deserialize(RUN_PATH)

    batchSizes      = runInfo.getBatchSizes()
    shapePipelineA  = runInfo.getMatrixShape(0)
    shapePipelineB  = runInfo.getMatrixShape(1)



    # Exit App
    sys.exit(0)