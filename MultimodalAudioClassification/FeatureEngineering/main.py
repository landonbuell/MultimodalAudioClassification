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
from sklearn.preprocessing import StandardScaler

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
    scaler1 = StandardScaler(copy=False)
    scaler1.fit(data1.getFeatures())
    scaler1.transform(data1.getFeatures())

    # Check that it worked
    means1 = data1.means()
    varis1 = data1.variances()

    # Create the other standard scaler
    scaler2 = Preprocessors.StandardScaler(numFeatures)
    scaler2.fit(data2)
    scaler2.call(data2)

    # Check that it worked
    means2 = data2.means()
    varis2 = data2.variances()

    # Exit App
    sys.exit(0)