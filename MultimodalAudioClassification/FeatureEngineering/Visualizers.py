"""
Repository:     MultimodalAudioClassification
Solution:       MultimodalAudioClassification
Project:        CommonToolsPy
File:           PyToolsStructures.py
 
Author:         Landon Buell
Date:           June 2022
"""

        #### IMPORTS ####

import os
import numpy as np
import matplotlib.pyplot as plt

import PyToolsStructures
import PyToolsIO

        #### CLASS DEFINITIONS ####

class BoxAndWhiskerPlots:
    """ Generate Box and Whisker Plots for Features in a pipeline """

    __NAME = "boxAndWhiskerPlots"

    def __init__(self,runInfo,pipelineIndexes):
        """ Constructor """
        self._runInfo = runInfo
        self._pipelineIndexes = pipelineIndexes
        self._outputPath = os.path.join(runInfo.getOutputPath(),BoxAndWhiskerPlots.__NAME)

    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    # Private Interface
