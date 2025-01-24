"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    FeautureEngineering
File:       main.py

Author:     Landon Buell
Date:       Sept 2022
"""

    #### IMPORTS ####

import sys
import os

import dataset

    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set some constants + Load Run Info
    INPUT_PATH  = "C:\\Users\\lando\\Documents\\audioFeatures\\simpleSignalsV1"
    data = dataset.Dataset(INPUT_PATH)
    spectrograms    = data.loadAllFromPipeline(5)
    


    sys.exit(0)