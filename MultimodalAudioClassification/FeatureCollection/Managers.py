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


import numpy as np
import pandas as pd

from scipy import signal

import PyToolsStructures

import Administrative
import CollectionMethods
import Structural

        #### CLASS DEFINITIONS ####

class Manager:
    """
    Manager is an Abstract Base Class in which all managers inherit from
    """

    def __init__(self):
        """ Constructor for Manager Base Class """
        self.logConstruction()

    def __del__(self):
        """ Destructor for Manager Base Class """
        self.logDestruction()

    # Getters and Setters

    def getSettings(self):
        """ Get a reference to the Runtime Settings from the getInstance() """
        return Administrative.FeatureCollectionApp.getInstance().getSettings()

    def getSampleManager(self):
        """ Get a reference to the Sample Manager form the getInstance() """
        return Administrative.FeatureCollectionApp.getInstance().getSampleManager()

    def getCollectionManager(self):
        """ Get a reference to the collection Manager from the getInstance() """
        return Administrative.FeatureCollectionApp.getInstance().getCollectionManager()

    def getRundataManager(self):
        """ Get a reference to the Rundata Manager from the the getInstance() """
        return Administrative.FeatureCollectionApp.getInstance().getRundataManager()

    # Public Interface

    def init(self):
        """ Initialize all Paramters for this Manager """
        self.logBuild()
        return self

    def clean(self):
        """ Cleanup the Manager """
        self.logCleanup()
        return self

    def logMessageInterface(self,msg,timeStamp=True):
        """ Simplified Interface for Logging Message via the CollectionApplicationPrototype """
        Administrative.FeatureCollectionApp.getInstance().logMessage(msg,timeStamp)
        return None

    # Protected Interface

    def describe(self):
        """ Log Description of the Current State of this Instance """
        msg = "Description of " + repr(self)
        self.logMessageInterface(msg,True)
        return self

    def logConstruction(self):
        """ Log Construction of Sample Manager """
        msg = "Constructing " + str(self.__class__) + " Instance..."
        self.logMessageInterface(msg)
        return None

    def logInit(self):
        """ Log the Initialization of the instance """
        msg = "Initializing " + str(self.__class__) + " Instance..."
        self.logMessageInterface(msg)
        return None

    def logCleanup(self):
        """ Log the Initialization of the instance """
        msg = "Cleaning " + str(self.__class__) + " Instance..."
        self.logMessageInterface(msg)
        return None

    def logDestruction(self):
        """ Log Construction of Sample Manager """
        msg = "Destroying " + str(self.__class__) + " Instance..."
        self.logMessageInterface(msg)
        return None

    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))


class SampleManager (Manager):
    """ SampleManager collects and organizes all data samples """

    def __init__(self):
        """ Constructor for SampleManager Instance """
        super().__init__()
        self._counter               = 0
        self._database              = np.array([],dtype=object)
        self._fileParserCallback    = None

    def __del__(self):
        """ Destructor for SampleManager Instance """
        super().__del__()
        
    # Getters and Setters

    def getDatabaseSize(self):
        """ Get the Size of the Database """
        return len(self._database)

    def getNextBatch(self):
        """ Get the next Batch of Samples """
        batchSize = self.getSettings().getBatchSize()
        batch = np.array(size=(batchSize,),dtype=object)
        
        # Get the next few items
        for i in range(0,batchSize):
            if (self._counter >= len(self)):
                # No samples left
                msg = "Last sample dequeued from batch database "
                Administrative.FeatureCollectionApp.getInstance().logMessage(msg)
                return batch

            # Otherwise....
            batch[i] = self._database[self._counter]
            self._counter += 1;

        # Batch is populated now
        return batch

    def samplesRemaining(self):
        """ Return T/F If there are no More samples to process """
        return (len(self._database) - self._counter)

    def registerFileParserCallback(self,callback):
        """ Register the method that will read a File into a list of samples """
        self._fileParserCallback = callback
        return self

    # Public Interface

    def init(self):
        """ Prepare the Sample Manager for Usage """
        super().init()

        # Register the callback to read the files
        self.registerFileParserCallback(
            SampleManager.createSamplesFromFile)

        # Read all Input Files
        self.readInputFiles()

        # Shuffle the Samples (If Applicable)
        self.shuffle()

        return self

    # Private Interface

    def readInputFiles(self):
        """ Read Through All Input Files and Add to Sample Database """
        inputFiles = self.getSettings().getInputPaths()
        samplesInFile = None
        
        # Visit Each Input File + Get All Samples
        for path in inputFiles:
            # Log this File
            msg = "Reading samples from file: {0}".format(path)
            self.logMessageInterface(msg)
            
            # Invoke the callback to parse the input file
            samplesInFile = self._fileParserCallback(path)
            self._database = np.append(self._database,samplesInFile)

            # Log Number of Samples
            msg = "\tFound {0} samples".format(samplesInFile.shape[0])
            self.logMessageInterface(msg)

        # Finished Parsing all Samples
        return self
          
    def shuffle(self):
        """ Shuffle Samples in Place According to Seed """
        seed = self.getSettings().getShuffleSeed()
        if (seed > -1):
            # Non-Negative Seed - Shuffle in place
            np.random.set_state(seed)
            self._database = np.random.shuffle(self._database)
        # If Seed is Negative, do not shuffle samples
        return self

    # Magic Methods

    def __len__(self):
        """ Overload Length Operator """
        return self._database.shape[0]

    def __iter__(self):
        """ Forward Iterator """
        for item in self._database:
            yield item

    # Static Methods

    @staticmethod
    def createSamplesFromFile(filePath):
        """ Read a file, and return an array of samples from it """
        frame = pd.read_csv(filePath,index_col=False)
        frame = frame.to_numpy()
        sampleArray = np.empty(shape=(frame.shape[0],),dtype=object)

        # Visit Each Sample in the Row
        for i,row in enumerate(frame):
            # Get Params from the row
            samplePath  = row[0]
            tgtInt      = int(row[1])
            tgtStr      = row[2]
            # Create the SampleIO Instance + Update Int -> Str Map
            sample = Structural.SampleIO(samplePath,tgtInt,tgtStr)
            # Add the Sample
            sampleArray[i] = sample

        return sampleArray

class RundataManager (Manager):
    """ RundataManager Aggregates all important info from the Collection run process """
    
    def __init__(self):
        """ Constructor for MetadataManager Instance """
        super().__init__()
        self._runInfo           = None

    def __del__(self):
        """ Destructor for MetadataManager Instance """
        super().__del__()

    # Getters and Setters

    def getRunInfo(self):
        """ Get RunInformation """
        return self._runInfo

    # Public Interface

    def init(self):
        """ Build the Data Manager Instance """
        super().init()
        self.getSettings().serialize()
             

        self.describe()      
        return self

    def processBatch(self,batchSamples,batchIndex):
        """ Process a batch of Samples """
        msg = "\tProcessing Batch {0}...".format(batchIndex)
        self.logMessageInterface(msg)

        # For each Sample in the Batch
        batchSize = len(batchSamples)
        for idx,sample in batchSamples:
            msg = "\t\tSample ({0:<4}/{1:<4})".format(idx,batchSize)
            self.logMessageInterface(msg)

            # Evaliate Feature Pipeline
            self.evaluatePipelines(sample,idx)

         
        # Update Run information
        self._runInfo.addBatchSize(batchSize)

        # All Done!
        return True


    def clean(self):
        """ Run Cleaning method on Instance """
        super().clean()
        
        # Serialize the run info 
        runInfoOutputPath = os.path.join(self.getSettings().getOutputPath() ,"runInfo.txt")
        self._runInfo.serialize(runInfoOutputPath)
        
        return self

    # Private Interface

    def evaluatePipelines(self,sample,sampleIndex):
        """ Evaluate Sample Against Each Feature Pipeline """

        pipelines = Administrative.FeatureCollectionApp.getPipelines()
        for pipeline in pipelines:

            features = pipeline.evaluate(sample)



        return 


class FeatureCollectionPipeline:
    """ Holds a Queue of Methods and Design Matrix """

    MAX_QUEUE_SIZE = 32

    def __init__(self,pipelineIdentifier):
        """ FeatureCollectionPipeline Constructor """
        self._identfier     = pipelineIdentifier
        self._initalized    = False
        self._queue         = np.array(size=(FeatureCollectionPipeline.MAX_QUEUE_SIZE,),dtype=object)  
        self._designMatrix  = None
        self._frameParams   = Structural.AnalysisFramesParameters()
        self._signalPreprocessCallbacks = []

    def __del__(self):
        """ FeatureCollectionPipeline Destructor """

    # Getters and Setters

    def getPipelineIdentifier(self):
        """ Get the name of this Pipeline """
        return self._identfier

    def getDesignMatrix(self):
        """ Get Design Matrix for this Pipeline """
        return self._designMatrix

    def getReturnSize(self):
        """ Get the Number of Features from this Pipeline """
        result = 0
        for item in self:
            result += item.getReturnSize()
        return result

    def getAnalysisFrameParams(self):
        """ Get the analysis frame parameters for this pipeline """
        return self._frameParams

    def setAnalysisFrameParams(self,params):
        """ Set the analysis frame parameters for this pipeline """
        self._frameParams = params
        return self

    def __getitem__(self,key):
        """ Return Item at Index """
        return self._queue[key]

    def __setitem__(self,key,val):
        """ Set Item at Index """
        self._queue[key] = val
        return self

    # Public Interface

    def registerSignalPreprocessCallback(self,callback):
        """ Register a Callback for preprocessing a signal """
        self._signalPreprocessCallbacks.append(callback)
        return self

    def initialize(self):
        """ Prepare the pipeline for processing """

    def evaluate(self,signal,sampleIndex):
        """ Evaluate Input signal against the queue of features """
        
        if(self._initalized == False):
            errMsg = "Must initialize FeatureCollectionPipeline {0} before use".format(self._identfier)
            Administrative.FeatureCollectionApp.getInstance().logMessage(msg)
            raise RuntimeError(errMsg)
                                                                                        
        # Prepare to Evaluate the Queue
        indexCounter = 0
        returnSize = self.getReturnSize()
        self.evaluateSignalPreprocessCallbacks(signal)
        featureVector = PyToolsStructures.FeatureVector(
            returnSize,
            signal.getClassLabel() )

        # Evaluate Queue
        for item in self:
            result = item.invoke(signal)

            # Put result into feature vector
            for j in range(result.shape[0]):
                featureVector[indexCounter] = result[j]
                indexCounter += 1

        # Return the Feature Vector from this Samples
        return featureVector
            
    def __iter__(self):
        """ Define Forward iterator """
        for i in range(FeatureCollectionPipeline.MAX_QUEUE_SIZE):
            if (self[i] == 0):
                continue
            yield self[i]

    # Private Interface

    def evaluateSignalPreprocessCallbacks(self,signal):
        """ Pass the signal through each callback """
        for callback in self._signalPreprocessCallbacks:
            callback(signal)
        return self

    # Static Interface

    @staticmethod
    def getDefaultPipelineAlpha():
        """ Default Pipeline Alpha """
        pipeline = FeatureCollectionPipeline("A")      

        # Register the CollectionMethods
        pipeline[0] = CollectionMethods.TimeDomainEnvelopPartitions(8)
        pipeline[1] = CollectionMethods.TimeDomainEnvelopFrames(0,256,32)
        pipeline[2] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.1)
        pipeline[3] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.2)
        pipeline[4] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.3)
        pipeline[5] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.4)
        pipeline[6] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.5)
        pipeline[7] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.6)
        pipeline[8] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.7)
        pipeline[9] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.8)
        pipeline[10] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.9)
        pipeline[11] = CollectionMethods.ZeroCrossingsPerTime(1)
        pipeline[12] = CollectionMethods.ZeroCrossingsFramesMean(1)
        pipeline[13] = CollectionMethods.ZeroCrossingsFramesVariance(1)
        pipeline[14] = CollectionMethods.ZeroCrossingsFramesDiffMinMax(1)
        pipeline[15] = CollectionMethods.TemporalCenterOfMass(1)
        pipeline[16] = CollectionMethods.TemporalCenterOfMass(3)
        pipeline[17] = CollectionMethods.AutoCorrelationCoefficients(12)
        pipeline[18] = CollectionMethods.AutoCorrelationCoefficientsMean(1)
        pipeline[19] = CollectionMethods.AutoCorrelationCoefficientsVariance(1)
        pipeline[20] = CollectionMethods.AutoCorrelationCoefficientsDiffMinMax(1)
        pipeline[21] = CollectionMethods.FrequencyCenterOfMass(1)
        pipeline[22] = CollectionMethods.FrequencyCenterOfMass(2)
        pipeline[23] = CollectionMethods.MelFilterBankEnergies(12)
        pipeline[24] = CollectionMethods.MelFilterBankEnergiesMean(1)
        pipeline[25] = CollectionMethods.MelFilterBankEnergiesVariance(1)
        pipeline[26] = CollectionMethods.MelFilterBankEnergiesDiffMinMax(1)
        pipeline[27] = CollectionMethods.MelFrequencyCepstrumCoefficients(12)

        # Return the resulting pipeline
        return pipeline

    @staticmethod
    def getDefaultPipelineBeta():
        """ Defulat Pipeline Alpha """
        pipeline = FeatureCollectionPipeline("B")

        # Register the CollectionMethods

        # Return the resulting pipeline
        return pipeline