"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureEngineering
    File:       dataset.py
    Classes:    DatasetLoader

    Author:     Landon Buell
    Date:       April 2024
"""

        #### IMPORTS ####

import os

        #### CLASS DEFINITIONS ####

class Dataset:
    """ 
        Represents a dataset that can be loaded for later use 
    """

    def __init__(self,
                 rootPath: str,
                 seed=123456789):
        """ Constructor """
        self._rootPath  = rootPath
        self._runInfo   = None
        self._classInfo = None
        self._pipelines = []

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    # Public Interface

    # Private Interface

    def __validateRootPath(self) -> None:
        """ Validate that the root path exists """
        if (os.path.isdir(self._rootPath) == False):
            msg = "Provided root path at {0} does not exist".format(self._rootPath)
            raise RuntimeError(msg)
        return None

    def __loadPipelines(self) -> None:
        """ Load in each pipeline dataset as a tf dataset """
        rootContents = os.listdir(self._rootPath)
        pipeline = "pipeline"
        for item in rootContents:
            if (pipeline not in item):
                continue
            pipelineRoot = os.path.join(self._rootPath,item)

            
