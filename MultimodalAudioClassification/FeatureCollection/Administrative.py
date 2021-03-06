"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureCollection  
File:           Administrative.py
 
Author:         Landon Buell
Date:           December 2021
"""

        #### IMPORTS ####

import os
import datetime

import PyToolsIO

import Managers


        #### CLASS DEFINITIONS ####

class FeatureCollectionApp:
    """ 
    Contains All Application Functions for FeatureCollection
    """

    # Static Memebers
    _appInstance = None

    def __init__(self,appSettings):
        """ Constructor for CollectionApplication Instance """
        FeatureCollectionApp._appInstance = self

        self._settings          = appSettings 
        self._logger            = Logger()
        
        self._sampleManager     = Managers.SampleManager()
        self._rundataManager    = Managers.RundataManager()

        self._pipelines         = [None,None]

        
    def __del__(self):
        """ Destructor for CollectionApplication Instance """
        self.logDestruction()

    @staticmethod
    def constructApp(settings):
        """ Construct the Application """
        if (FeatureCollectionApp._appInstance is None):
            FeatureCollectionApp._appInstance = FeatureCollectionApp(settings)
        else:
            errMsg = "Can only have one instance of FeatureCollectionApp at runtime"
            raise RuntimeError(errMsg)
        return FeatureCollectionApp._appInstance

    @staticmethod
    def destroyApp():
        """ Destroy the Aplication """
        if (FeatureCollectionApp._appInstance is not None):
            FeatureCollectionApp._appInstance = None
        return True
     
    # Getters and Setters

    @staticmethod
    def getInstance():
        """ Return the application Instance if it exists """
        if (FeatureCollectionApp._appInstance is None):
            # App Does not Exist
            errMsg = "ERROR: FeatureCollectionApp has not been instantiated"
            raise RuntimeError(errMsg)
        return FeatureCollectionApp._appInstance

    def getSettings(self):
        """ Return the Settings Instance """
        return self._settings

    def getLogger(self):
        """ Return the logger Instance """
        return self._logger

    def getSampleManager(self):
        """ Return the Sample Manager """
        return self._sampleManager

    def getRundataManager(self):
        """ Return the Data Manager """
        return self._rundataManager

    def getPipelines(self):
        """ Get Feature Pipeline at Index """
        return self._pipelines

    # Public Interface

    def startup(self):
        """ Run Application Startup Sequence """
        
        self._sampleManager.init()
        self._rundataManager.init()

        # Emplace Feature Pipelines
        self._pipelines[0] = Managers.FeatureCollectionPipeline.getDefaultPipelineAlpha()
        self._pipelines[1] = Managers.FeatureCollectionPipeline.getDefaultPipelineBeta()

        # Initialize the Pipelines
        self._pipelines[0].initialize()
        self._pipelines[1].initialize()

        return self

    def execute(self):
        """ Run Application Execution Sequence """
        
        batchCounter = 0
        MAX_BATCHES = 100000
        loop = True

        while (batchCounter < MAX_BATCHES and loop == True):

            # Get the Next Batch
            batch = self.getSampleManager().getNextBatch()
            loop = self.getRundataManager().processBatch(batch,batchCounter)
            
            if (self.getSampleManager().samplesRemaining() <= 0):
                # Not more samples to process
                loop = False

            # Otherwise....
            # TODO: This
            batchCounter += 1
            
        # All Done!
        msg = "\t Last batch Processed!"
        self.logMessage(msg)
        return self

    def shutdown(self):
        """ Run Application Shutdown Sequence """

        self._sampleManager.clean()
        self._rundataManager.clean()


        return self
    
    # Internal Interface

    def logMessage(self,message,timeStamp=True):
        """ Log Message To User """
        self._logger.logMessage(message,timeStamp)
        return self

    def checkIterationConditions(self):
        """ Return T/F - if """

    @staticmethod
    def getDateTime() -> str:
        """ Get formatted DateTime as String """
        result = str(datetime.datetime.now())
        result = result.replace("-",".")
        result = result.replace(":",".")
        result = result.replace(" ",".")
        return result

    def logConstruction(self):
        """ Log Construction of Sample Manager """
        msg = "Constructing FeatureCollectionApp Instance ..."
        FeatureCollectionApp._appInstance.logMessage(msg)
        return None

    def logDestruction(self):
        """ Log Construction of Sample Manager """
        msg = "Destroying FeatureCollectionApp Instance ..."
        FeatureCollectionApp._appInstance.logMessage(msg)
        return None

    # Magic Methods

    def __repr__(self):
        """ Debugger representation of Instance """
        if (FeatureCollectionApp._appInstance is None):
            # Not Yet Initialized
            return "No Instance"
        else:
            memAddress = str(hex(id(FeatureCollectionApp._appInstance)))
            return str(self.__class__) + " @ " + memAddress

class AppSettings:
    """
    Contains all runtime settings for duration of application
    """
    def __init__(self,pathsInput,pathOutput,batchSize=32,batchLimit=-1,shuffleSeed=-1):
        """ Constructor for AppSettings Instance """
        self._pathStartup   = os.getcwd()
        self._pathsInput    = set()
        self._pathOutput    = None
        self._batchSize     = batchSize
        self._batchLimit    = batchLimit
        self._shuffleSeed   = shuffleSeed
        self._verbose       = 1
        self._logToConsole  = True
        self._logToFile     = True

        self.initInputPaths(pathsInput)
        self.initOutputPath(pathOutput)

    def __del__(self):
        """ Destructor for AppSettings Instance """
        pass

    # Getters and Setters

    def getStartupPath(self) -> str:
        """ Get Application Startup Path """
        return self._pathStartup

    def getInputPaths(self) -> set:
        """ Return List of Input Paths """
        return self._pathsInput

    def getOutputPath(self) -> str:
        """ Return Output Path """
        return self._pathOutput

    def getBatchSize(self) -> int:
        """ Return the Batch Size """
        return self._batchSize

    def getBatchLimit(self) -> int:
        """ Return the batch counter limit """
        return self._batchLimit

    def getShuffleSeed(self) -> int:
        """ Return the Sufffle Seed """
        return self._shuffleSeed

    def getVerbose(self) -> bool:
        """ Return T/F if in Verbose Mode """
        return self._verbose

    def getLogToConsole(self) -> bool:
        """ Get T/F If Log to Console """
        return self._logToConsole

    def getLogToFile(self) -> bool:
        """ Get T/F IF Log to File """
        return self._logToFile

    # Public Interface

    def addInputPath(self,path) -> bool:
        """ Add New Input Path to the Set of Paths """
        fullPath = os.path.abspath(path)
        self._pathsInput.add(fullPath)
        return self

    def serialize(self,outputPath=None)-> bool:
        """ Write the Settings Instance out to a text file """
        if (outputPath is None):
            outputPath = os.path.join(self.getOutputPath(),"settings.txt")
        writer = PyToolsIO.AppSettingsSerializer(self,outputPath)
        success = True
        try:
            writer.call()
        except Exception as err:
            print("\t\tAppSettings.serialize() " + err)
            success = False
        return success

    @staticmethod
    def developmentSettingsInstance():
        """ Build an instance of runtime settings for development """
        result = AppSettings(
            pathsInput=[],
            pathOutput="..\\..\\..\\..\\audioFeatures\\simpleSignalsV0",
            batchSize=64,
            batchLimit=-1,
            shuffleSeed=-1)
        #result.addInputPath("..\\..\\InputFiles\\Y1.csv")
        #result.addInputPath("..\\..\\InputFiles\\Y2.csv")
        #result.addInputPath("..\\..\\InputFiles\\Y3.csv")
        result.addInputPath("..\\..\\InputFiles\\Y4.csv")
        return result

    # Private Interface

    def initInputPaths(self,pathSet):
        """ Initialize Set of Input Paths """
        for x in pathSet:
            self.addInputPath(x)
        return self

    def initOutputPath(self,output):
        """ Initialize the Output Path """
        fullOutput = os.path.abspath(output)
        if (os.path.isdir(fullOutput)):
            # Content may be overwritten
            msg = "WARNING: Output path exists. Contents may be over written"
            # Cannot Log Message yet - no app instance
        else:
            os.makedirs(fullOutput)
        self._pathOutput = fullOutput
        return self

    # Magic Methods

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))
            

class Logger:
    """ 
    Handles all runtime Logging 
    """

    def __init__(self):
        """ Constructor for Logger Instance """      
        self._path          = None
        self._outFile       = None
        self._toConsole     = FeatureCollectionApp._appInstance.getSettings().getLogToConsole()
        self._toFile        = FeatureCollectionApp._appInstance.getSettings().getLogToFile()
        
        if (self._toFile):
            self._outFile = open("logger.txt","w")
        self.writeHeader()

    def __del__(self):
        self.writeFooter()
        """ Destructor for Logger Instance """
        if (self._outFile is not None):
            if (self._outFile.closed() == False):
                self._outFile.close()
        self._outFile = None

    # Getters and Setters

    def getLoggerPath(self):
        """ Return the Path to the logger text output file """
        outpath = FeatureCollectionApp._appInstance.getSettings().getOutputPath()
        return os.path.join(outpath,"logger.txt")

    # Public Interface

    def logMessage(self,message:str,timeStamp=True):
        """ Log Message to Console or Text """
        formattedMessage = None
        if (timeStamp == True):
            # Log Message w/ a TimeStamp
            now = FeatureCollectionApp.getDateTime()
        else:
            # Log Message w/o a TimeStamp
            now = ""
        formattedMessage = "\t{0:<32}\t{1:<128}".format(now,message)

        # Write the Message to Console and/or to File
        if (self._toConsole == True):
            print(formattedMessage)

        if (self._toFile == True):
            self._outFile = open(self.getLoggerPath(),"a")
            self._outFile.write(formattedMessage)
            self._outFile.close()
        return self

    # Private Interface

    def writeHeader(self):
        """ Write Header To Logger """
        header = [
            self.spacer(),
            "FeatureCollectionApp",
            FeatureCollectionApp.getDateTime(),
            self.spacer()
            ]
        # Log Each Line of the Header
        for msg in header:
            self.logMessage(msg,False)
        return self

    def writeFooter(self):
        """ Write Footer To Logger """
        footer = [
            self.spacer(),
            "FeatureCollectionApp",
            FeatureCollectionApp.getDateTime(),
            self.spacer()
            ]
        # Log Each Line of the Header
        for msg in footer:
            self.logWithoutTimeStamp(msg)
        return self

    def spacer(self,numChars=64):
        """ Get a Spacer String """
        return "\n" + ("-" * numChars) + "\n"
    