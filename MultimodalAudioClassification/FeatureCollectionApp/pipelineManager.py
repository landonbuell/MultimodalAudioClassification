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

import os

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
        self._featurePipelines  = []
        self._outputStreams     = [None] * 10        
        self._rowFmt = "{0:<16}{1:<16}{2:<16}{3}\n"

    def __del__(self):
        """ Destructor """
        super().__del__()
        self.__closeAllOutputStreams()

    # Accessors

    def getNumPipelines(self) -> int:
        """ Return the number of feature pipelines """
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
        return None

    def teardown(self) -> None:
        """ OVERRIDE: Teardown the Sample Database """
        super().teardown()
        self.__closeAllOutputStreams()
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
        featureVectors  = [None] * len(self._featurePipelines)
        successFlags    = [None] * len(self._featurePipelines)
        for ii,pipeline in enumerate(self._featurePipelines):
            featureVectors[ii]  = pipeline.evaluate(signal)
        self.__exportSampleFileData(signal)
        return featureVectors
            
    # Private Interface

    def __exportPipelineInfo(self) -> None:
        """ Export info about each pipeline to their appropriate paths """
        MAX_NUM_NAMES_ALLOWED = 10000
        for pipeline in self._featurePipelines:
            if (pipeline.getManager() != self):
                pipeline.setManager(self)         
            if (pipeline.getNumFeatures() < MAX_NUM_NAMES_ALLOWED):
                pipeline.exportFeatureNames()         
            pipeline.exportFeatureShapes()
        return None

    def __exportSampleFileData(self,
                               signal: signalData.SignalData) -> None:
        """ Log that the pipeline manager processed a sample """
        sampleFileIndex = signal.uniqueID() // self.getSettings().getSamplesPerOutputFile()
        sampleFileIndexStr = str(sampleFileIndex)
        if (sampleFileIndex < 10):
            sampleFileIndexStr = "0" + sampleFileIndexStr

        # Locate the output stream to send the row data to
        if (sampleFileIndex >= len(self._outputStreams)):
            for ii in range(len(self._outputStream),sampleFileIndex):
                self._outputStreams.append(None)

        # Now see if we need to create the file there
        if (self._outputStreams[sampleFileIndex] is None):
            # create the stream
            streamPath = os.path.join(
                self.getSettings().getOutputPath(),
                "samples{0}.txt".format(sampleFileIndexStr))

            self._outputStreams[sampleFileIndex] = open(streamPath,"w")
            headerText = self._rowFmt.format(
                "sampleIndex","classIndex","channelIndex","sourcePath")
            self._outputStreams[sampleFileIndex].write(headerText)

        # The stream is now open, write to it
        rowText = self._rowFmt.format(
            signal.uniqueID(),
            signal.getTarget(),
            signal.getChannelIndex(),
            signal.getSourcePath())
        self._outputStreams[sampleFileIndex].write(rowText)
        return None

    def __closeAllOutputStreams(self) -> None:
        """ Close all output streams """
        for stream in self._outputStreams:
            if (stream is not None):
                stream.close()
        self._outputStreams.clear()
        return None





