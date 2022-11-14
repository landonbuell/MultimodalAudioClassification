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

import PyToolsStructures
import Preprocessors

import numpy as np


    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set some constants + Load Run Info
    RUN_PATH = "C:\\Users\\lando\\Documents\\audioFeatures\\simpleSignalsV1"
    runInfo = PyToolsStructures.RunInfo.deserialize(RUN_PATH)

    # Load Batches
    allBatchesA = runInfo.loadAllBatches(True,False)
    numFeatures = allBatchesA[0].getNumFeatures()

    data1 = allBatchesA[0].deepCopy()
    data2 = allBatchesA[0].deepCopy()

    # Create the Scaler
    scaler = Preprocessors.StandardScaler(numFeatures)
    scaler.fit(data1)
    scaler.call(data1)

    means = np.mean(data1.getFeatures(),axis=0)
    varis = np.var(data1.getFeatures(),axis=0)


    sys.exit(0)