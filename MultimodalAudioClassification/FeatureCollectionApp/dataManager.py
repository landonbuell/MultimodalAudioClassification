"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       dataManager.py
    Classes:    RundataManager

    Author:     Landon Buell
    Date:       April 2024
"""

        #### IMPORTS ####

import os

import componentManager
import sessionInformation

        #### CLASS DEFINITIONS ####

class RundataManager(componentManager.ComponentManager):
    """ Stores high-level data related to the current run """

    __NAME = "RundataManager"

    def __init__(self,
                 app):
        """ Constructor """
        super().__init__(RundataManager.__NAME,app)
        self._runInfo   = sessionInformation.RunInfo()
        self._classInfo = sessionInformation.ClassInfoDatabase()

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getRunInfo(self) -> object:
        """ Return a ref to the run info structure """
        return self._runInfo

    def getClassInfo(self) -> object:
        """ Return a ref to the class info structure """
        return self._classInfo

    def getExportLocation(self,
                      pipelineIndex: int,
                      classIndex: int) -> str:
        """ Return the export location (but not file name) for a sample """
        pipelinePath = self.getApp().getPipelineManager().getOutputPath(pipelineIndex)
        classPath = "class{0}".format(classIndex)
        return os.path.join(pipelinePath,classPath)

    # Public Interface

    def initialize(self) -> None:
        """ OVERRIDE: Initialize the Sample Database """
        super().initialize()
        self.__initRunInfo()
        self.__initClassInfo()
        self.__initPipelineOutputPaths()
        return None

    def teardown(self) -> None:
        """ OVERRIDE: Teardown the Sample Database """
        super().teardown()
        self.__exportRunInfo()
        self.__exportClassInfo()
        return None

    def registerExpectedSample(self, sampleData: object) -> None:
        """ Register this sample w/ the data manager """
        sampleTargetIndex   = sampleData.getTarget()    
        if (self._classInfo.hasClassIndex(sampleTargetIndex) == False):
            sampleTargetName    = "UNKNOWN_CLASS_NAME"
            self._classInfo.registerClass(sampleTargetIndex,sampleTargetName)
            msg = "Adding class#{0} to expected samples".format(sampleTargetIndex)
            self.logMessage(msg)
        self._classInfo.incrementExpectedCount(sampleTargetIndex)
        self.__initClassOutputPaths(sampleTargetIndex)
        return None

    def registerProcessedSample(self, signalData: object) -> None:
        """ Register a processed sample w/ the data manager """  
        sampleTargetIndex   = signalData.getTarget()    
        if (self._classInfo.hasClassIndex(sampleTargetIndex) == False):
            msg = "Got class Index of {0} which does NOT exist in the ClassInfoDatabase".format(
                sampleTargetIndex)
            self.logMessage(msg)
            raise RuntimeError(msg)
        self._classInfo.incrementProcessedCount(sampleTargetIndex)
        return None

    def registerExportedSample(self, featureVector: object) -> None:
        """ Register a processed sample w/ the data manager """  
        sampleTargetIndex   = featureVector.getTarget()    
        if (self._classInfo.hasClassIndex(sampleTargetIndex) == False):
            msg = "Got class Index of {0} which does NOT exist in the ClassInfoDatabase".format(
                sampleTargetIndex)
            self.logMessage(msg)
            raise RuntimeError(msg)
        self._classInfo.incrementExportedCount(sampleTargetIndex)
        return None

    # Private Interface

    def __initRunInfo(self) -> None:
        """ Initialize the run info structure """
        return None

    def __initClassInfo(self) -> None:
        """ Initialize the class Info Struct """
        return None

    def __initPipelineOutputPaths(self) -> None:
        """ Create the top-level output path for each pipeline """
        numPipelines = self.getApp().getPipelineManager().getNumPipelines()
        for ii in range(numPipelines):
            pipelinePath = self.getApp().getPipelineManager().getOutputPath(ii)
            if (os.path.isdir(pipelinePath) == False):
                # Path does NOT exist
                msg = "Making pipeline export path: {0}".format(pipelinePath)
                self.logMessage(msg)
                os.mkdir(pipelinePath)
        return None
   
    def __initClassOutputPaths(self,classIndex: int) -> None:
        """ Create the top-level output path for each class within each pipeline """
        numPipelines = self.getApp().getPipelineManager().getNumPipelines()
        for ii in range(numPipelines):
            fullPath = self.getExportLocation(ii,classIndex)
            if (os.path.isdir(fullPath) == False):
                # Path does NOT exist
                msg = "Making pipeline export path: {0}".format(fullPath)
                self.logMessage(msg)
                os.mkdir(fullPath)
        return None

    def __exportRunInfo(self) -> None:
        """ Export the run info struct to the output path """
        return None

    def __exportClassInfo(self) -> None:
        """ Export the class info struct to the output path """
        outputFile = os.path.join(self.getSettings().getOutputPath(),"classInfo.txt")
        self._classInfo.exportToFile(outputFile)
        return None
