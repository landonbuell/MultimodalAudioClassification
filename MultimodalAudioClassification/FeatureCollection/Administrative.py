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
import PyToolsStructures

import Managers


        #### CLASS DEFINITIONS ####

class FeatureCollectionApp:
    """ 
    Contains All Application Functions for FeatureCollection
    """

    # Static Memebers
    __appInstance = None

    def __init__(self,appSettings):
        """ Constructor for CollectionApplication Instance """
        FeatureCollectionApp.__appInstance = self

        self._settings          = appSettings 
        self._logger            = Logger.initFromApp(self)
        
        self._sampleManager     = Managers.SampleManager()
        self._rundataManager    = Managers.RundataManager()

        self._pipelines         = [None] * PyToolsStructures.RunInformation.DEFAULT_NUM_PIPELINES

        
    def __del__(self):
        """ Destructor for CollectionApplication Instance """
        self.__logDestruction()
        
    @staticmethod
    def constructApp(settings):
        """ Construct the Application """
        if (FeatureCollectionApp.__appInstance is None):
            FeatureCollectionApp.__appInstance = FeatureCollectionApp(settings)
        else:
            errMsg = "Can only have one instance of FeatureCollectionApp at runtime"
            raise RuntimeError(errMsg)
        return FeatureCollectionApp.__appInstance
     
    # Getters and Setters

    @staticmethod
    def getInstance():
        """ Return the application Instance if it exists """
        if (FeatureCollectionApp.__appInstance is None):
            # App Does not Exist
            errMsg = "ERROR: FeatureCollectionApp has not been instantiated"
            raise RuntimeError(errMsg)
        return FeatureCollectionApp.__appInstance

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
        
        # Emplace Feature Pipelines
        self._pipelines[0] = Managers.FeatureCollectionPipeline.getDefaultPipelineAlpha()
        self._pipelines[1] = Managers.FeatureCollectionPipeline.getDefaultPipelineBeta()

        # Initialize the Pipelines
        self._pipelines[0].initialize()
        self._pipelines[1].initialize()

        # Initialize the Collection Managers
        self._sampleManager.init()
        self._rundataManager.init()

        return self

    def execute(self):
        """ Run Application Execution Sequence """
        
        batchCounter = 0
        batchLimit = self.getSettings().getBatchLimit()
        loop = True

        while (loop == True):

            # Get the Next Batch
            batch = self.getSampleManager().getNextBatch()
            loop = self.getRundataManager().processBatch(batch,batchCounter)
            
            if (self.getSampleManager().samplesRemaining() <= 0):
                # Not more samples to process
                msg = "Processed all Samples ..."
                self.logMessage(msg)
                loop = False

            if (batchCounter >= AppSettings.MAX_BATCHES - 1):
                # Max Number of batches reached
                msg = "batchCounter exceeded MAX_BATCHES of {0} ...".format(AppSettings.MAX_BATCHES )
                self.logMessage(msg)
                loop = False

            if (batchCounter >= batchLimit - 1):
                # Max Number of batches reached
                msg = "batchCounter exceeded batchLimit of {0} ...".format(batchLimit)
                self.logMessage(msg)
                loop = False

            # Otherwise, incremenet + procceed
            batchCounter += 1
            
        # All Done!
        msg = "\tLast batch Processed!"
        self.logMessage(msg)
        return self

    def shutdown(self):
        """ Run Application Shutdown Sequence """

        self._sampleManager.clean()
        self._rundataManager.clean()

        return self
    
    # Protected Interface

    def logMessage(self,message,timeStamp=True):
        """ Log Message To User """
        self._logger.logMessage(message,timeStamp)
        return self

    @staticmethod
    def getDateTime() -> str:
        """ Get formatted DateTime as String """
        result = str(datetime.datetime.now())
        result = result.replace("-",".")
        result = result.replace(":",".")
        result = result.replace(" ",".")
        return result

    # Private Interface

    def __logConstruction(self):
        """ Log Construction of Sample Manager """
        msg = "Constructing FeatureCollectionApp Instance ..."
        FeatureCollectionApp.__appInstance.logMessage(msg)
        return None

    def __logDestruction(self):
        """ Log Construction of Sample Manager """
        msg = "Destroying FeatureCollectionApp Instance ..."
        self.logMessage(msg)
        return None

    # Magic Methods

    def __repr__(self):
        """ Debugger representation of Instance """
        if (FeatureCollectionApp.__appInstance is None):
            # Not Yet Initialized
            return "No Instance"
        else:
            memAddress = str(hex(id(FeatureCollectionApp.__appInstance)))
            return str(self.__class__) + " @ " + memAddress

class AppSettings:
    """
    Contains all runtime settings for duration of application
    """

    MAX_BATCHES = 100000

    def __init__(self,pathsInput,pathOutput,batchSize=32,batchLimit=-1,sampleLimit=100000,shuffleSeed=-1):
        """ Constructor for AppSettings Instance """
        self._pathStartup   = os.getcwd()
        self._pathsInput    = set()
        self._pathOutput    = None
        self._batchSize     = batchSize
        self._batchLimit    = batchLimit
        self._sampleLimit   = sampleLimit
        self._shuffleSeed   = shuffleSeed
        self._verbose       = 1
        self._logToConsole  = True
        self._logToFile     = True

        self.__initInputPaths(pathsInput)
        self.__initOutputPath(pathOutput)

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

    def getSampleLimit(self) -> int:
        """ Return the sample processing limit """
        return self._sampleLimit

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
        writer = AppSettings.__AppSettingsSerializer(self,outputPath)
        writer.call()
        return True

    @staticmethod
    def developmentSettingsInstance():
        """ Build an instance of runtime settings for development """
        result = AppSettings(
            pathsInput=[],
            pathOutput="..\\..\\..\\..\\audioFeatures\\simpleSignalsV1",
            batchSize=16,
            batchLimit=256,
            shuffleSeed=-1)
        #result.addInputPath("..\\..\\InputFiles\\Y1.csv")
        #result.addInputPath("..\\..\\InputFiles\\Y2.csv")
        #result.addInputPath("..\\..\\InputFiles\\Y3.csv")
        result.addInputPath("..\\..\\InputFiles\\Y4.csv")
        return result

    # Private Interface

    def __initInputPaths(self,pathSet):
        """ Initialize Set of Input Paths """
        for x in pathSet:
            self.addInputPath(x)
        return self

    def __initOutputPath(self,output):
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

    class AppSettingsIOKeys:
        """ Keys used in read/write AppSetting instance to disk """
        KEY_PATH_STARTUP        = "PathStartup"
        KEY_PATH_INPUT          = "PathInput"
        KEY_PATH_OUTPUT         = "PathOutput"
        KEY_BATCH_SIZE          = "BatchSize"
        KEY_BATCH_LIMIT         = "BatchLimit"
        KEY_SAMPLE_LIMIT        = "SampleLimit"
        KEY_SHUFFLE_SEED        = "ShuffleSeed"
        KEY_VERBOSITY           = "Verbosity"
        KEY_LOG_TO_CONSOLE      = "LogToConsole"
        KEY_LOG_TO_FILE         = "LogToFile"

    class __AppSettingsSerializer(PyToolsIO.Serializer):
        """ Class to Help Serialize App Settings Instance """

        def __init__(self,data,path):
            """ Constructor """
            super().__init__(data,path)

        def __del__(self):
            """ Destructor """
            super().__del__()

        # Public Interface

        def call(self):
            """ Run the Serializer """
            super().call()
            self.__writePaths()
            self._writeFooter()
            return True


        # Private Interface

        def __writePaths(self):
            """ Write Input + Output + Startup paths """                
            line = PyToolsIO.Serializer.fmtKeyValPair(
                AppSettings.AppSettingsIOKeys.KEY_PATH_STARTUP,
                self._data.getStartupPath() )
            self.appendLine(line)

            for ii,path in enumerate(self._data.getInputPaths()):
                line =  PyToolsIO.Serializer.fmtKeyValPair(
                    AppSettings.AppSettingsIOKeys.KEY_PATH_INPUT + "[{0}]".format(ii),
                    path )
                self.appendLine(line)

            line =  PyToolsIO.Serializer.fmtKeyValPair(
                AppSettings.AppSettingsIOKeys.KEY_PATH_OUTPUT,
                self._data.getOutputPath() )
            self.appendLine(line)

            return self

        def __writeBatchData(self):
            """ Write Out data related to batches """
            line = PyToolsIO.Serializer.fmtKeyValPair(
                AppSettings.AppSettingsIOKeys.KEY_BATCH_SIZE,
                self._data.getBatchSize() )
            self.appendLine( line )

            line = PyToolsIO.Serializer.fmtKeyValPair(
                AppSettings.AppSettingsIOKeys,KEY_BATCH_LIMIT,
                self._data.getBatchLimit() )
            self.appendLine( line )

            line = PyToolsIO.Serializer.fmtKeyValPair(
                 AppSettings.AppSettingsIOKeys,KEY_SAMPLE_LIMIT,
                self._data.getSampleLimit() )
            self.appendLine( line )

            line = PyToolsIO.Serializer.fmtKeyValPair(
                AppSettings.AppSettingsIOKeys,KEY_SHUFFLE_SEED,
                self._data.getShuffleSeed() )
            self.appendLine( line )

            return self

        def __writeLoggerData(self):
            """" Write out data related to logging """
            line = PyToolsIO.Serializer.fmtKeyValPair(
                AppSettings.AppSettingsIOKeys,KEY_VERBOSITY,
                self._data.getVerbose() )
            self.appendLine( line )

            line = PyToolsIO.Serializer.fmtKeyValPair(
                AppSettings.AppSettingsIOKeys,KEY_LOG_TO_CONSOLE,
                self._data.getLogToConsole() )
            self.appendLine( line )

            line = PyToolsIO.Serializer.fmtKeyValPair(
                AppSettings.AppSettingsIOKeys,KEY_LOG_TO_FILE,
                self._data.getLogToFile() )
            self.appendLine( line )

            return self

    class __AppSettingsDeserializer(PyToolsIO.Deserializer):
        """ Cllas to Help Deserialize App Settings Instance """

        def __init__(self,path):
            """ Constructor """
            super().__init__(self,path)

        def __del__(self):
            """ Destructor """
            super().__del__()

        # Private Interface

    # Magic Methods

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))
            

class Logger:
    """ 
    Handles all runtime Logging 
    """

    def __init__(self,outpath,toConsole=True,toFile=True):
        """ Constructor for Logger Instance """ 
        self._outPath       = os.path.join(outpath,"logger.txt")
        self._outFile       = None
        self._toConsole     = toConsole
        self._toFile        = toFile
        
        if (self._toFile):
            self._outFile = open(self._outPath,"w")
        self.__writeHeader()

    def __del__(self):
        self.__writeFooter()
        """ Destructor for Logger Instance """
        if (self._outFile is not None):
            if (self._outFile.closed() == False):
                self._outFile.close()
        self._outFile = None

    @staticmethod
    def initFromApp(app):
        """ Construct from Feature Collection App Instance """
        outpath     = app.getSettings().getOutputPath()
        toConsole   = app.getSettings().getLogToConsole()
        toFile      = app.getSettings().getLogToFile()
        logger = Logger(outpath,toConsole,toFile)
        return logger

    # Getters and Setters

    def getLoggerPath(self):
        """ Return the Path to the logger text output file """
        return self._outPath

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
        formattedMessage = "\t{0:<32}\t{1}".format(now,message)

        # Write the Message to Console and/or to File
        if (self._toConsole == True):
            print(formattedMessage)

        if (self._toFile == True):
            self._outFile = open(self.getLoggerPath(),"a")
            self._outFile.write(formattedMessage + "\n")
            self._outFile.close()
        return self

    # Private Interface

    def __writeHeader(self):
        """ Write Header To Logger """
        header = [
            self.__spacer(),
            "FeatureCollectionApp",
            FeatureCollectionApp.getDateTime(),
            self.__spacer()
            ]
        # Log Each Line of the Header
        for msg in header:
            self.logMessage(msg,False)
        return self

    def __writeFooter(self):
        """ Write Footer To Logger """
        footer = [
            self.__spacer(),
            "FeatureCollectionApp",
            FeatureCollectionApp.getDateTime(),
            self.__spacer()
            ]
        # Log Each Line of the Header
        for msg in footer:
            self.logMessage(msg,False)
        return self

    def __spacer(self,numChars=64):
        """ Get a Spacer String """
        return "\n" + ("-" * numChars) + "\n"
    