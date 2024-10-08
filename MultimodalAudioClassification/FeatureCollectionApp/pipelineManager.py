"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       componentManager.py
    Classes:    ComponentManager

    Author:     Landon Buell
    Date:       February 2024
"""

        #### IMPORTS ####

import componentManager
import featurePipeline
import signalData
import coreCallbacks

        #### CLASS DEFINITIONS ####

class PipelineManager(componentManager.ComponentManager):
    """ Stores a collection of samples to process """

    __NAME = "PipelineManager"

    def __init__(self,
                 app):
        """ Constructor """
        super().__init__(PipelineManager.__NAME,app)
        self._featurePipelines = []

    def __del__(self):
        """ Destructor """
        super().__del__()
        self._featurePipelines.clear()

    # Accessors

    def getSize(self) -> int:
        """ Return the current size of the database """
        return len(self._featurePipelines)

    def getOutputPath(self,pipelineIndex: int) -> str:
        """ Return the output path for the pipeline at the provided index """
        return self._featurePipelines[pipelineIndex].getOutputPath()

    # Public Interface

    def initialize(self) -> None:
        """ OVERRIDE: Initialize the Sample Database """
        super().initialize()
        self.registerPipeline( coreCallbacks.DefaultFeaturePipeline.getDefaultPipeline00() )
        self.registerPipeline( coreCallbacks.DefaultFeaturePipeline.getDefaultPipeline01() )        
        self.registerPipeline( coreCallbacks.DefaultFeaturePipeline.getDefaultPipeline02() )        
        self.registerPipeline( coreCallbacks.DefaultFeaturePipeline.getDefaultPipeline03() )
        self.__exportPipelineInfo()
        self.__exportFeatureNames()
        return None

    def teardown(self) -> None:
        """ OVERRIDE: Teardown the Sample Database """
        super().teardown()
        return None

    def registerPipeline(self,
                         pipeline: featurePipeline.FeaturePipeline) -> None:
        """ Add a feature pipeline to the manager """
        msg = "Registerng pipeline: {0} w/ PipelineManager".format(
            pipeline.getName())
        self.logMessage(msg)
        # Register
        self._featurePipelines.append(pipeline)
        pipeline.setManager(self)
        return None

    def processSignal(self,
                      signal: signalData.SignalData) -> list:
        """ Evalute signal on each pipeline & return list of feature vectors """
        featureVectors = [None] * len(self._featurePipelines)
        for ii,pipeline in enumerate(self._featurePipelines):
            featureVectors[ii] = pipeline.evaluate(signal)
        return featureVectors
            
    # Private Interface

    def __exportPipelineInfo(self) -> None:
        """ Export info about each pipeline to their appropriate paths """
        return None

    def __exportFeatureNames(self) -> None:
        """ Export all feature names for a single pipeline """
        for pipeline in self._featurePipelines:
            pipeline.exportFeatureNames()
        return None

            





