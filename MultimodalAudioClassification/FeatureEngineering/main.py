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

import PyToolsStructures
import Preprocessors

import numpy as np


    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set some constants + Load Run Info
    INPUT_PATH = "C:\\Users\\lando\\Documents\\audioFeatures\\simpleSignalsV1"
    OUTPUT_PATH = "C:\\Users\\lando\\Documents\\audioFeatures\\simpleSignalsV1_processed"
    runInfo = PyToolsStructures.RunInfo.deserialize(INPUT_PATH)

    # Load Batches
    allBatches = runInfo.loadAllBatches(True,False)
    numFeaturesA = allBatches[0].getNumFeatures()
    #numFeaturesB = allBatches[1].getNumFeatures()

    # Create the Scaler
    scaler = Preprocessors.StandardScaler(numFeaturesA)
    scaler.fit(allBatches[0])
    scaler.call(allBatches[0])

    # Show that It worked
    means = np.mean(allBatches[0].getFeatures(),axis=0)
    varis = np.var(allBatches[0].getFeatures(),axis=0)

    # Write The Params for the scaler
    scaler.serialize( os.path.join(OUTPUT_PATH,"paramsStandardScaler.txt") )
    sys.exit(0)