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
from typing import Callable

import numpy as np
import pandas as pd

import PyToolsStructures
import PyToolsIO

import Administrative
import CollectionMethods
import Structural
import Callbacks

        #### CLASS DEFINITIONS ####

class Manager:
    """
    Manager is an Abstract Base Class in which all managers inherit from
    """

    def __init__(self):
        """ Constructor for Manager Base Class """
        self._logConstruction()

    def __del__(self):
        """ Destructor for Manager Base Class """
        self._logDestruction()

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
        self._logInit()
        return self

    def clean(self):
        """ Cleanup the Manager """
        self._logCleanup()
        return self

    def describe(self):
        """ Log Description of the Current State of this Instance """
        msg = "Description of " + repr(self)
        self._logMessageInterface(msg,True)
        return self
    
    # Protected Interface

    def _logMessageInterface(self,msg,timeStamp=True):
        """ Simplified Interface for Logging Message via the CollectionApplicationPrototype """
        Administrative.FeatureCollectionApp.getInstance().logMessage(msg,timeStamp)
        return None

    def _logConstruction(self):
        """ Log Construction of Sample Manager """
        msg = "Constructing " + str(self.__class__) + " Instance..."
        self._logMessageInterface(msg)
        return None

    def _logInit(self):
        """ Log the Initialization of the instance """
        msg = "Initializing " + str(self.__class__) + " Instance..."
        self._logMessageInterface(msg)
        return None

    def _logCleanup(self):
        """ Log the Initialization of the instance """
        msg = "Cleaning " + str(self.__class__) + " Instance..."
        self._logMessageInterface(msg)
        return None

    def _logDestruction(self):
        """ Log Construction of Sample Manager """
        msg = "Destroying " + str(self.__class__) + " Instance..."
        self._logMessageInterface(msg)
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
        self._classData             = PyToolsStructures.CategoryDatabase()
        self._fileParserCallback    = None

    def __del__(self):
        """ Destructor for SampleManager Instance """
        super().__del__()
        
    # Getters and Setters

    def getDatabaseSize(self):
        """ Get the Size of the Database """
        return len(self._database)

    def getClassDatabase(self):
        """ Get the Info for the processed classes """
        return self._classData

    def getNextBatch(self):
        """ Get the next Batch of Samples """
        batchSize = self.getSettings().getBatchSize()
        batch = np.array([],dtype=object)
        
        # Get the next few items
        for i in range(0,batchSize):
            if (self._counter >= len(self)):
                # No samples left
                msg = "Last sample dequeued from batch database "
                Administrative.FeatureCollectionApp.getInstance().logMessage(msg)
                return batch

            # Otherwise....
            batch = np.append(batch, self._database[self._counter])
            self._counter += 1
            
        # Batch is populated now, add to data
        self._classData.updateWithBatchData(batch)
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
        self.__readInputFiles()
        self.__shuffle()
        self.describe()
        return self

    def clean(self):
        """ OVERRIDE : Cleanup the Sample Manager after usage """
        classDataOutputPath = os.path.join(self.getSettings().getOutputPath(),"classData.txt")
        self._classData.serialize(classDataOutputPath)

        return None

    # Private Interface

    def __readInputFiles(self):
        """ Read Through All Input Files and Add to Sample Database """
        inputFiles = self.getSettings().getInputPaths()
        samplesInFile = None
        
        # Visit Each Input File + Get All Samples
        for path in inputFiles:
            # Log this File
            msg = "Reading samples from file: {0}".format(path)
            self._logMessageInterface(msg)
            
            # Invoke the callback to parse the input file
            samplesInFile = self._fileParserCallback(path)
            self._database = np.append(self._database,samplesInFile)

            # Log Number of Samples
            msg = "\tFound {0} samples".format(samplesInFile.shape[0])
            self._logMessageInterface(msg)

        # Update RunInfo w/ Expected num samples
        sampleDatabaseSize = len(self._database)
        runInfo = Administrative.FeatureCollectionApp.getInstance().getRundataManager().getRunInfo()
        runInfo.setExpectedNumberOfSamples(sampleDatabaseSize)
        return self
          
    def __shuffle(self):
        """ Shuffle Samples in Place According to Seed """
        seed = self.getSettings().getShuffleSeed()
        if (seed < 0):
            return self    
        # Non-Negative Seed - Shuffle in place
        np.random.seed(seed)
        np.random.shuffle(self._database)
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

class RundataManager(Manager):
    """ RundataManager Aggregates all important info from the Collection run process """
    
    def __init__(self):
        """ Constructor for MetadataManager Instance """
        super().__init__()
        inputPaths = Administrative.FeatureCollectionApp.getInstance().getSettings().getInputPaths()
        outputPath = Administrative.FeatureCollectionApp.getInstance().getSettings().getOutputPath()
        self._runInfo           = PyToolsStructures.RunInformation(inputPaths,outputPath)

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
        self.__registerPipelines()
        self.describe()      
        return self

    def processBatch(self,batchSamples,batchIndex):
        """ Process a batch of Samples """
        msg = "Processing Batch {0}...".format(batchIndex)
        self._logMessageInterface(msg)

        # Realloc for Batch Size
        batchSize = len(batchSamples)
        self.__reallocDesignMatrices(batchSize)

        # For each Sample in the Batch
        for i in range(batchSize):
            msg = "\tSample ({0:<2}/{1:<2})".format(i,batchSize)
            self._logMessageInterface(msg)

            # Evaliate Feature Pipeline
            signal = batchSamples[i].readSignal()
            self.__evaluatePipelines(signal,i)

         
        # Update Run information
        self._runInfo.addBatchSize(batchSize)
        self.__exportDesignMatrices(batchIndex)

        # All Done!
        return True


    def clean(self):
        """ Run Cleaning method on Instance """
        super().clean()
 
        # Store that classes that were processed
        classesInUse = self.getSampleManager().getClassDatabase().getClassesInUse()
        self._runInfo.setClassesInUse(classesInUse)

        # Serialize the run info 
        runInfoOutputPath = os.path.join(self.getSettings().getOutputPath() ,"runInfo.txt")
        self._runInfo.serialize(runInfoOutputPath)
        
        return self

    # Private Interface

    def __registerPipelines(self):
        """ Register Each Pipeline w/ the Run Info """
        pipelines = Administrative.FeatureCollectionApp.getInstance().getPipelines()
        for ii,pipeline in enumerate(pipelines):
            if (pipeline is None):
                continue
            self._runInfo.registerPipeline(pipeline,ii)
        return self

    def __evaluatePipelines(self,signal,sampleIndex):
        """ Evaluate Sample Against Each Feature Pipeline """

        pipelines = Administrative.FeatureCollectionApp.getInstance().getPipelines()
        for pipeline in pipelines:
            if (pipeline is None):
                continue
            features = pipeline.evaluate(signal)
            pipeline.getDesignMatrix().__setitem__(sampleIndex,features)
       
        # Done!
        return 

    def __reallocDesignMatrices(self,batchSize):
        """ Reallocate Design Matrices if different batch Size """
        pipelines = Administrative.FeatureCollectionApp.getInstance().getPipelines()
        for pipeline in pipelines:
            if (pipeline is None):
                continue
            numSamples = pipeline.getDesignMatrix().getNumSamples()
            # check if different size
            if (numSamples != batchSize):
                pipeline.getDesignMatrix().setNumSamples(batchSize)
        # Done reallocatinging all design Matrices
        return self

    def __exportDesignMatrices(self,batchIndex):
        """ Export design Matrices """
        pipelines = Administrative.FeatureCollectionApp.getInstance().getPipelines()
        
        # Export X + Y for pipeline 0 and Export only X for pipline 1
        pipelines[0].exportDesignMatrix(batchIndex,True,True)
        pipelines[1].exportDesignMatrix(batchIndex,True,False)
       
        return self

class FeatureCollectionPipeline:
    """ Holds a Queue of Methods and Design Matrix """

    MAX_QUEUE_SIZE = 32

    def __init__(self,pipelineIdentifier):
        """ FeatureCollectionPipeline Constructor """
        self._identfier     = pipelineIdentifier
        self._initalized    = False
        self._size          = 0
        self._queue         = np.zeros(shape=(FeatureCollectionPipeline.MAX_QUEUE_SIZE,),dtype=object)  
        self._designMatrix  = None
        self._frameParams   = Structural.AnalysisFramesParameters()
        self._signalPreprocessCallbacks = []
        self._featureVectorPostProcessCallbacks = []

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
        if (key >= self._size or key < 0):
            msg = "Index: {0} is out of range for FeatureCollectionPipeline w/ size {1}".format(
                key,self._size)
            raise RuntimeError(msg)
        return self._queue[key]

    def __setitem__(self,key,val):
        """ Set Item at Index """
        if (key >= self._size or key < 0):
            msg = "Index: {0} is out of range for FeatureCollectionPipeline w/ size {1}".format(
                key,self._size)
            raise RuntimeError(msg)
        self._queue[key] = val
        return self

    # Public Interface

    def registerCollectionMethod(self,method):
        """ Register a Feature Collection method with this pipeline """
        if (self._size >= FeatureCollectionPipeline.MAX_QUEUE_SIZE):
            msg = "FeatureCollectionPipeline has reached max capactity. Cannot subscribe new methods."
            RuntimeError(msg)
        # Add the methof to the queue
        self._queue[self._size] = method
        self._size += 1
        return self

    def registerSignalPreprocessCallback(self,callback):
        """ Register a Callback for preprocessing a signal """
        self._signalPreprocessCallbacks.append(callback)
        return self

    def registerFeatureVectorPostprocessCallback(self,callback):
        """ Register a Callback for post processing a signal """
        self._featureVectorPostProcessCallbacks.append(callback)
        return self

    def resize(self,newSize):
        """ Resize the Queue to fit specific size """
        if (newSize < 1 or newSize > FeatureCollectionPipeline.MAX_QUEUE_SIZE):
            msg = "New size of {0} is invalid, but be between (1,32)".format(newSize)
            raise RuntimeError(msg)
        # Resize
        self._size = newSize
        return self

    def initialize(self):
        """ Prepare the pipeline for processing """

        returnSize = (self.getReturnSize(),)
        batchSize = Administrative.FeatureCollectionApp.getInstance().getSettings().getBatchSize()
        self._designMatrix = PyToolsStructures.DesignMatrix(batchSize,returnSize)

        # Register this pipeline instance as the 'owner' of each method
        for item in self:
            item.registerWithPipeline(self)

        # Mark Initialization Finished
        self._initalized = True
        return self

    def evaluate(self,signal):
        """ Evaluate Input signal against the queue of features """
        
        if(self._initalized == False):
            errMsg = "Must initialize FeatureCollectionPipeline {0} before use".format(self._identfier)
            Administrative.FeatureCollectionApp.getInstance().logMessage(errMsg)
            raise RuntimeError(errMsg)
                                                                                        
        # Prepare to Evaluate the Queue
        indexCounter = 0
        returnSize = self.getReturnSize()
        self.__evaluateSignalPreprocessCallbacks(signal)
        featureVector = PyToolsStructures.FeatureVector(
            sampleShape=(returnSize,),
            label=signal.getTargetInt() )
        expectedSize = 1

        # Evaluate Queue (Iterate through collection methods)
        for collector in self:
            expectedSize    = collector.getReturnSize()
            result          = collector.invoke(signal)

            if (result.shape[0] != expectedSize):
                # Shape Mismatch
                msg = "Expected retrun size {0} but got {1} from {2}".format(
                    expectedSize,result.shape[0],collector)
                Administrative.FeatureCollectionApp.getInstance().logMessage(msg)
                raise RuntimeError(msg)

            # Put result into feature vector
            for item in result:
                featureVector[indexCounter] = item
                indexCounter += 1

        # Return the Feature Vector from this Samples
        self.__evaluateFeatureVectorPostprocessCallbacks(featureVector)
        return featureVector

    def exportDesignMatrix(self,batchIndex,exportX=True,exportY=True):
        """ Export the Design Matrices to Disk """
        exportPath = Administrative.FeatureCollectionApp.getInstance().getSettings().getOutputPath()
        pathX,pathY = None,None
        if (exportX == True):
            pathX = PyToolsStructures.getDesignMatrixDataPath(exportPath,batchIndex,self.getPipelineIdentifier())
        if (exportY == True):
            pathY = PyToolsStructures.getDesignMatrixLabelsPath(exportPath,batchIndex)
        success = self._designMatrix.serialize(pathX,pathY)
        return success
            
    def __iter__(self):
        """ Define Forward iterator """
        for ii in range(self._size):
            if (self._queue[ii] == 0):
                continue        # Skip if nothing there
            yield self._queue[ii]

    # Private Interface

    def __evaluateSignalPreprocessCallbacks(self,signal):
        """ Pass the signal through each callback """
        for callback in self._signalPreprocessCallbacks:
            callback.__call__(self,signal)
        return self
    
    def __evaluateFeatureVectorPostprocessCallbacks(self,featureVector):
        """ Pass the signal through each callback """
        for callback in self._featureVectorPostProcessCallbacks:
            callback.__call__(self,featureVector)
        return self

    # Static Interface

    @staticmethod
    def getDefaultPipelineAlpha():
        """ Default Pipeline Alpha """
        pipeline = FeatureCollectionPipeline("A")      
        pipeline.resize( 32 )

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
        #pipeline[11] = CollectionMethods.ZeroCrossingsPerTime(1)
        #pipeline[12] = CollectionMethods.ZeroCrossingsFramesMean(1)
        #pipeline[13] = CollectionMethods.ZeroCrossingsFramesVariance(1)
        #pipeline[14] = CollectionMethods.ZeroCrossingsFramesDiffMinMax(1)
        pipeline[15] = CollectionMethods.TemporalCenterOfMass("linear")
        pipeline[16] = CollectionMethods.TemporalCenterOfMass("natural_log")
        pipeline[17] = CollectionMethods.AutoCorrelationCoefficients(12)
        pipeline[18] = CollectionMethods.AutoCorrelationCoefficientsMean(1)
        pipeline[19] = CollectionMethods.AutoCorrelationCoefficientsVariance(1)
        pipeline[20] = CollectionMethods.AutoCorrelationCoefficientsDiffMinMax(1)
        pipeline[21] = CollectionMethods.FrequencyCenterOfMassMean("linear")
        pipeline[22] = CollectionMethods.FrequencyCenterOfMassVari("linear")
        pipeline[23] = CollectionMethods.FrequencyCenterOfMassMean("natural_log")
        pipeline[24] = CollectionMethods.FrequencyCenterOfMassVari("natural_log")
        pipeline[25] = CollectionMethods.MelFilterBankEnergies(12)
        pipeline[26] = CollectionMethods.MelFilterBankEnergiesMean(1)
        pipeline[27] = CollectionMethods.MelFilterBankEnergiesVariance(1)
        pipeline[28] = CollectionMethods.MelFilterBankEnergiesDiffMinMax(1)
        pipeline[29] = CollectionMethods.MelFrequencyCepstrumCoefficients(12)

        # Return the resulting pipeline
        pipeline.registerSignalPreprocessCallback( Callbacks.SignalDataPreprocessCallbacks.makeAnalysisFramesTime )
        pipeline.registerSignalPreprocessCallback( Callbacks.SignalDataPreprocessCallbacks.makeAnalysisFramesFreq )
     
        return pipeline

    @staticmethod
    def getDefaultPipelineBeta():
        """ Defualt Pipeline Beta """
        pipeline = FeatureCollectionPipeline("B")
        pipeline.resize( 1 )
        # Register the CollectionMethods
        
        pipeline[0] = CollectionMethods.Spectrogram(
            pipeline.getAnalysisFrameParams() )

        #pipeline.registerFeatureVectorPostprocessCallback( Callbacks.FeatureVectorPostProcessCallbacks.plotSpectrogram )

        # Return the resulting pipeline
        return pipeline
