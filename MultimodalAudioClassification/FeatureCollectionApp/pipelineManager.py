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

        #### CLASS DEFINITIONS ####

class PipelineManager(componentManager.ComponentManager):
    """ Stores a collection of samples to process """

    __NAME = "PipelineManager"

    def __init__(self,
                 app: componentManager.featureCollectionApp.FeatureCollectionApplication):
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
        self._size

    # Public Interface

    def initialize(self) -> None:
        """ OVERRIDE: Initialize the Sample Database """
        super().initialize()
        self.registerPipeline( featurePipeline.DefaultFeaturePipeline.getDefaultPipeline00() )
        self.registerPipeline( featurePipeline.DefaultFeaturePipeline.getDefaultPipeline01() )
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






