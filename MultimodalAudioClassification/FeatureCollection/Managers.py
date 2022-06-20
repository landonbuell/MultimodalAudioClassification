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
import sys
import numpy as np
import pandas as pd
from scipy import signal

import CommonPlotting

import Administrative
import CollectionMethods
import Structural

import CommonStructures

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

    def getRuntimeSettings(self):
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

    def build(self):
        """ Initialize all Paramters for this Manager """
        self.logBuild()
        return self

    def call(self):
        """ Run the Execution of this Manager """
        self.logExecution()
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

    def logBuild(self):
        """ Log the Initialization of the instance """
        msg = "Initializing " + str(self.__class__) + " Instance..."
        self.logMessageInterface(msg)
        return None

    def logExecution(self):
        """ Log the Initialization of the instance """
        msg = "Executing " + str(self.__class__) + " Instance..."
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
        self._sampleDataBase    = np.array([],dtype=object)
        self._batchSizes        = None
        self._sampleIndex       = 0

    def __del__(self):
        """ Destructor for SampleManager Instance """
        super().__del__()
        
    # Getters and Setters

    def getSample(self,idx):
        """ Get Sample From Database at Index """
        return self._sampleDataBase[idx]

    def setSample(self,idx,sample):
        """ Set Sample to Database at Index """
        self._sampleDataBase[idx] = sample
        return self

    def getBatchSizes(self):
        """ Get Array of Each Batch Size """
        return self._batchSizes

    def getNumSamples(self) -> int:
        """ Get the Total Number of Samples """
        return self._sampleDataBase.shape[0]

    def getNumBatches(self) -> int:
        """ Get the Number of Batches in this Run """
        return self._batchSizes.shape[0]

    def getSizeOfBatch(self, batchIndex: int) -> int:
        """ Get the Size of the i-th batch """
        if (batchIndex >= self.getNumBatches()):
            errMsg = "Batch Index is out of range"
            raise ValueError(errMsg)
        return self._batchSizes[batchIndex]

    def getNextSample(self):
        """ Get the Sample Pointed to by the Index """
        if (self._sampleIndex >= self.getNumSamples()):
            # Reset the Sample Counter
            sampleIndex = 0
        else:
            result = self._sampleDataBase[self._sampleIndex]
            self._sampleDataBase[self._sampleIndex] = 0
            self._sampleIndex += 1
        return result

    # Public Interface

    def describe(self):
        """ Export Description of this Instance """
        super().describe()
        messages = [
            "Total samples: {0}".format(len(self)),
            "Number of classes: {0}".format(self.getRundataManager().getNumClasses()),
            "Number of batches: {0}".format(self.getNumBatches())
            ]
        for msg in messages:
            # Log Each String as a Message
            self.logMessageInterface(msg,False)
        return self

    def build(self):
        """ Gather + Organize all Audio Samples """
        super().build()
        self.readInputFiles()
        self.createSizeOfEachBatch()
        self.shuffle()
        self.describe()
        return self

    def createBatch(self,batchIndex: int):
        """ Get an Array of Samples for the Next Batch """
        # Create the Batch Subset
        batchSize = self.getSizeOfBatch(batchIndex)
        indexStart = batchIndex * batchSize
        batch = np.empty(shape=(batchSize,),dtype=object)
        
        # Populate Batch w/ Entries from Database
        for i in range(batchSize):
            batch[i] = self.getNextSample()

        return batch

    # Private Interface

    def readInputFiles(self):
        """ Read Through All Input Files and Add to Sample Database """
        inputFiles = self.getRuntimeSettings().getInputPaths()
        samplesInFile = None
        
        # Visit Each Input File + Get All Samples
        for path in inputFiles:
            # Log this File
            msg = "Reading samples from file: {0}".format(path)
            self.logMessageInterface(msg)
            # Get the Data
            samplesInFile = self.createSamplesFromFile(path)
            self._sampleDataBase = np.append(self._sampleDataBase,samplesInFile)
            # Log Number of Samples
            msg = "\tFound {0} samples".format(samplesInFile.shape[0])
            self.logMessageInterface(msg)

        return self

    def createSamplesFromFile(self,filePath):
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

    def createSizeOfEachBatch(self):
        """ Build a List for the Size of Each Batch """
        standardBatchSize = Administrative.FeatureCollectionApp.getInstance().getSettings().getBatchSize()
        numSamples = self._sampleDataBase.shape[0]
        numBatches = (numSamples // standardBatchSize)
        allBatchSizes = np.ones(shape=(numBatches,),dtype=int) * standardBatchSize
        extraSamples =  (numSamples % standardBatchSize)
        # Computer the Number of Batches (Include )
        if (extraSamples != 0):
            # There are "Extra" Samples
            allBatchSizes = np.append(allBatchSizes,extraSamples)
        self._batchSizes = allBatchSizes
        return self
          
    def shuffle(self):
        """ Shuffle Samples in Place According to Seed """
        seed = Administrative.FeatureCollectionApp.getInstance().getSettings().getShuffleSeed()
        if (seed <= -1):
            # Negative seed - do not shuffle
            return self
        # Non-Negative Seed - Shuffle in place
        np.random.set_state(seed)
        self._sampleDataBase = np.random.shuffle(self._sampleDataBase)
        return self

    # Magic Methods

    def __len__(self):
        """ Overload Length Operator """
        return self._sampleDataBase.shape[0]

    def __iter__(self):
        """ Forward Iterator """
        for item in self._sampleDataBase:
            yield item

    def __getitem__(self,key):
        """ Get Item at Index """
        return self._sampleDataBase[key]

    def __setitem__(self,key,val):
        """ Set Item at Index """
        self._sampleDataBase[key] = val
        return self

    
class CollectionManager (Manager):
    """ CollectionManager organizes all Features Methods """

    def __init__(self):
        """ Constructor for CollectionManager Instance """
        super().__init__()
        self._batchIndex        = 0
        self._batchQueue        = np.array([],dtype=object)
        self._methodQueueA      = np.array([],dtype=object)    
        self._methodQueueB      = np.array([],dtype=object) 
        self._callbacks         = []
        self._designMatrixA     = None
        self._designMatrixB     = None
        
    def __del__(self):
        """ Destructor for CollectionManager Instance """
        self._batchQueue            = None
        self._methodQueueA          = None
        self._methodQueueB          = None
        self._designMatrixA         = None
        self._designMatrixB         = None
        super().__del__()

    # Getters and Setters

    def getBatchIndex(self) -> int:
        """ Get the Current Batch Index """
        return self._batchIndex

    def getBatchQueue(self):
        """ Get the Current Batch Queue of Audio Files """
        return self._batchQueue

    def getMethodQueue(self):
        """ Get the Method Queue for the Collector """
        return self._methodQueueA

    def getDesignMatrixA(self):
        """ Get the Design Matrix A"""
        return self._designMatrixA

    def getDesignMatrixB(self):
        """ Get the Design Matrix B"""
        return self._designMatrixB

    def addCallback(self,callback):
        """ Add Callback to This instance """

    # Public Interface

    def build(self):
        """ Build All Data for Feature Collection """
        super().build()
        self.initCollectionQueue()
        self.initDesignMatrix()
        return self

    def call(self,batchIndex,batchSize):
        """ The Run the Collection Manager """
        super().call()

        # Log this Batch
        self.logCurrentBatch(batchIndex,batchSize)
        self._batchIndex = batchIndex

        # Build the Design Matrix
        shapeA = self.getRundataManager().getShapeSampleA()
        shapeB = self.getRundataManager().getShapeSampleB()
        self._designMatrixA = CommonStructures.DesignMatrix(batchSize,shapeA)
        self._designMatrixB = CommonStructures.DesignMatrix(batchSize,shapeB)

        # Create + Evaluate the Batch
        self.createBatchQueue(batchIndex)      
        self.evaluateBatchQueue()            

        # Psuedo-Callbacks?

        return self

    def clean(self):
        """ Final Cleanup on the Collection Manager """
        return super().clean()

    # Private Interface

    def initCollectionQueue(self):
        """ Build All Elements in the Collection Queue """
        numEntries = 32
        self._methodQueueA = np.zeros(shape=(numEntries,),dtype=object)
        # Populate with Elements
        self[0] = CollectionMethods.TimeDomainEnvelopPartitions(8)
        self[1] = CollectionMethods.TimeDomainEnvelopFrames(0,256,32)
        self[2] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.1)
        self[3] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.2)
        self[4] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.3)
        self[5] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.4)
        self[6] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.5)
        self[7] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.6)
        self[8] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.7)
        self[9] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.8)
        self[10] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.9)
        self[11] = CollectionMethods.ZeroCrossingsPerTime(1)
        self[12] = CollectionMethods.ZeroCrossingsFramesMean(1)
        self[13] = CollectionMethods.ZeroCrossingsFramesVariance(1)
        self[14] = CollectionMethods.ZeroCrossingsFramesDiffMinMax(1)
        self[15] = CollectionMethods.TemporalCenterOfMass(1)
        self[16] = CollectionMethods.TemporalCenterOfMass(3)
        self[17] = CollectionMethods.AutoCorrelationCoefficients(12)
        self[18] = CollectionMethods.AutoCorrelationCoefficientsMean(1)
        self[19] = CollectionMethods.AutoCorrelationCoefficientsVariance(1)
        self[20] = CollectionMethods.AutoCorrelationCoefficientsDiffMinMax(1)
        self[21] = CollectionMethods.FrequencyCenterOfMass(1)
        self[22] = CollectionMethods.FrequencyCenterOfMass(2)
        self[23] = CollectionMethods.MelFilterBankEnergies(12)
        self[24] = CollectionMethods.MelFilterBankEnergiesMean(1)
        self[25] = CollectionMethods.MelFilterBankEnergiesVariance(1)
        self[26] = CollectionMethods.MelFilterBankEnergiesDiffMinMax(1)
        self[27] = CollectionMethods.MelFrequencyCepstrumCoefficients(12)
        self[28] = 0
        self[29] = 0
        self[30] = 0
        self[31] = 0
        return self

    def initDesignMatrix(self):
        """ Initialize the Design Matrix Instance """
        numSamples = self.getSampleManager().getSizeOfBatch(self._batchIndex)
        shapeA = self.getRundataManager().getShapeSampleA()
        shapeB = self.getRundataManager().getShapeSampleB()
        self._designMatrixA = CommonStructures.DesignMatrix(numSamples, shapeA)
        self._designMatrixB = CommonStructures.DesignMatrix(numSamples, shapeB)
        return self

    def createBatchQueue(self,idx):
        """ Create the Current Batch of Samples """      
        self._batchQueue = Administrative.FeatureCollectionApp.getInstance().getSampleManager().createBatch(idx)
        return self

    def evaluateBatchQueue(self):
        """ Iterate through Batch Queue """
        signalData      = None

        # Build the Feature Vectors for Each Sample
        shapeA = self.getRundataManager().getShapeSampleA()
        shapeB = self.getRundataManager().getShapeSampleB()
        featureVectorA   = CommonStructures.FeatureVector(shapeA)
        featureVectorB   = CommonStructures.FeatureVector(shapeB)
            
        # Iterate Through Each Sample
        for idx,sample in enumerate(self._batchQueue):

            # Log this Sample
            self.logCurrentSample(idx,len(self._batchQueue))
            self.getRundataManager().getOccuranceData().updateActual(sample.getTargetInt())
            
            # Set the Label + Read the Raw Samples
            featureVectorA.setLabel(sample.getTargetInt())
            featureVectorB.setLabel(sample.getTargetInt())
            
            # Read Raw samples + Make Analysis Frames
            signalData = sample.readSignal()
            signalData.makeAnalysisFramesTime( 
                self.getRundataManager().getFrameParams() )
            signalData.makeAnalysisFramesFreq(
                self.getRundataManager().getFrameParams() )
            # Generate Each Common Field as needed by method

            # Use Current Sample to Evaluate the Feature Queue
            self.evaluateMethodQueueA(signalData,featureVectorA)
            self.evaluateMethodQueueB(signalData,featureVectorB)

            # Add to Samples to Design Matrices
            self._designMatrixA[idx] = featureVectorA
            self._designMatrixB[idx] = featureVectorB
            featureVectorA.clearData()
            featureVectorB.clearData()
            self.getRundataManager().getFrameParams().reset()

        # Update the Batch's Meta Data       
        batchDataA = Structural.BatchData(self.getBatchIndex(), self._designMatrixA)
        batchDataB = Structural.BatchData(self.getBatchIndex(), self._designMatrixB)

        self.getRundataManager().addBatchData(batchDataA,increment=True)
        self.getRundataManager().addBatchData(batchDataB,increment=False)
        
        signalData = None
        return self

    def evaluateMethodQueueA(self,signalData,featureVector):
        """ Evaluate the Feature Queue """
        featureIndex = 0
        result = None
        for item in self._methodQueueA:

            # Handle Empty Slot
            if (item == 0):             
                continue

            # Evalue the current method
            result = item.invoke(signalData)

            # Copy Result to the feature vector
            for i in range(item.getReturnSize()):
                featureVector[featureIndex] = result[i]
                featureIndex += 1
            result = None

        # Sanity Check
        assert(featureIndex == featureVector.getShape()[0])
        return self

    def evaluateMethodQueueB(self,signalData,featureVector):
        """ Evaluate the Feature Queue """
        featureIndex = 0
        result = None
        if(signalData.AnalysisFramesFreq is None):
            # No Analysis Frames Freq? -> Make them
            signalData.makeAnalysisFramesFreq(
                self.getRundataManager().getFrameParams() )

        # Input for CNN is the Freq-Series AnalysisFrames
        # Put the Freq-Series Analysis Frames into the 
        featureVector.setData( signalData.AnalysisFramesFreq, enforceShape=False )
       
        """
        # Check Contents of Feature Vector
        freq = signalData.FrequencyAxis
        time = np.arange(0,256,1)
        shapeSpectrogram = self.getRundataManager().getFrameParams().getFreqFramesShape()
        CommonPlotting.plotSpectrogram(
            featureVector.getData().reshape(shapeSpectrogram),
            time,freq,signalData._className)
        """
        return self

    def makeAllFields(self,signalData,framesTime=True,framesFreq=True,
                      MFCCs=True,ACCs=True,ZXRs=True,
                      energyTimeFrames=True,energyFreqFrames=True):
        """ Make all Required fields for Feature Extraction """
        if (framesTime == True):
            # Make Time-Series Analysis Frames
            signalData.makeAnalysisFramesTime( self.getRundataManager().getFrameParams() )
        if (framesFreq == True):
            # Make Freq-Series Analysis Frames
            signalData.makeAnalysisFramesFreq()
        if (MFCCs == True):
            # Make Mel-Freq Cepstrum Coeffs
            signalData.makeMelFreqCepstrumCoeffs()
        if (ACCs == True):
            # Make the Auto-Correlation Coeffcicients
            signalData.makeAutoCorrelationCoeffs()
        if (ZXRs == True):
            # Make the Zero-Crossing Rate Methods
            signalData.makeZeroCrossingRate()
        if (energyTimeFrames == True):
            # Make the Energy in Each Time-Frame
            signalData.makeFrameEnergiesTime()
        if (energyFreqFrames == True):
            # Make the Energy in Each Freq-Frame
            signalData.makeFrameEnergiesFreq()
        return self

    def logCurrentBatch(self,index,size):
        """" Log Current Batch w/ Num Samples """
        numBatches = Administrative.FeatureCollectionApp.getInstance().getSampleManager().getNumBatches()
        msg = "Running batch ({0}/{1}), with {2} samples".format(index+1,numBatches,size)
        self.logMessageInterface(msg)
        return None

    def logCurrentSample(self,index,size):
        """ Log Current Sample in Batch """
        msg = "\tProcessing sample ({0}/{1})".format(index+1,size)
        self.logMessageInterface(msg)
        return None

    # Magic Methods

    def __len__(self):
        """ Overload Length Operator """
        return self._methodQueueA.shape[0]

    def __getitem__(self,key):
        """ Get Item at index """
        return self._methodQueueA[key]

    def __setitem__(self,key,val):
        """ Set Item at Index """
        self._methodQueueA[key] = val
        return self

class RundataManager (Manager):
    """ RundataManager Aggregates all important info from the Collection run process """
    
    def __init__(self):
        """ Constructor for MetadataManager Instance """
        super().__init__()
        self._runInfo           = None
        self._batchDataObjs     = []
        self._occuranceData     = CommonStructures.ClassOccuranceData()
        self._frameParams       = None

    def __del__(self):
        """ Destructor for MetadataManager Instance """
        super().__del__()

    # Getters and Setters

    def getRunInfo(self):
        """ Get RunInformation """
        return self._runInfo

    def getOccuranceData(self):
        """ Get the occurance of each class """
        return self._occuranceData

    def getFrameParams(self):
        """ Return AnalysisFrameParameters Structure """
        return self._frameParams

    def getShapeSampleA(self):
        """ Get the Shape of the Samples in Data A """
        size = 0
        for item in self.getCollectionManager().getMethodQueue():
            if (item == 0):
                continue
            size += item.getReturnSize()
        return (size,)

    def getShapeSampleB(self,sampleRate=44100):
        """ Get the Shape of the Samples in Data B """
        if (self._frameParams is None):
            self.initAnalysisFrameParams()
        return self.getFrameParams().getFreqFramesShape()

    def getNumClasses(self):
        """ Get the Number of Unqiue Classes """
        return len(self._occuranceData.getUniqueClassInts())


    # Public Interface

    def build(self):
        """ Build the Data Manager Instance """
        super().build()
        self.getRuntimeSettings().serialize()
             
        self._runInfo = CommonStructures.RunInformation( 
            self.getRuntimeSettings().getInputPaths(),
            self.getRuntimeSettings().getOutputPath() )

        self.initSampleShapeSizes()
        self.initBatchSizeData()
        self.initAnalysisFrameParams()
        self.initFeatureNamesMatrixA()
        self.initSampleOccuranceData()

        self.describe()
        
        return self

    def call(self):
        """ Run this Manager's Execution Method """

        super().call()
        
        # Serialize the Design Matrix
        batchIndex = self.getCollectionManager().getBatchIndex()
        outputPath = Administrative.FeatureCollectionApp.getInstance().getSettings().getOutputPath()
        
        outXa   = os.path.join(outputPath,"batch{0}_Xa.bin".format(batchIndex))
        outXb   = os.path.join(outputPath,"batch{0}_Xb.bin".format(batchIndex))
        outY    = os.path.join(outputPath,"batch{0}_Y.bin".format(batchIndex))

        # Write the Data Out
        self.getCollectionManager().getDesignMatrixA().serialize(outXa,outY)
        self.getCollectionManager().getDesignMatrixB().serialize(outXb,None)

        # Compute Meta Data and then Clear
        self.getCollectionManager().getDesignMatrixA().clearData()
        self.getCollectionManager().getDesignMatrixB().clearData()

        return self

    def clean(self):
        """ Run Cleaning method on Instance """
        runOutputFolder = Administrative.FeatureCollectionApp.getInstance().getSettings().getOutputPath()     
        self._runInfo.serialize(os.path.join(runOutputFolder,"runInformation.txt"),
            Administrative.FeatureCollectionApp.getInstance().getSettings().getBatchLimit())
        self._occuranceData.serialize(os.path.join(runOutputFolder,"classData.txt"))
        super().clean()
        return self

    def addBatchData(self,batchData,increment):
        """ Add Batch Data Instance to this Instance """
        self._batchDataObjs.append(batchData)
        if (increment == True):
            self._runInfo.incrementActualNumSamples( batchData.getNumSamples() )
        return self

    # Private Interface

    def describe(self):
        """ Log description of state of this instance """
        super().describe()
        msg = "{0:<16}{1:<32}{2:<16}".format("Int","Name","Total")
        self.logMessageInterface(msg,False)
        for items in self.getOccuranceData():
            msg = "{0:<16}{1:<32}{2:<16}".format(items[0],items[1],items[3])
            self.logMessageInterface(msg,False)
        # Return
        return self

    def initSampleShapeSizes(self):
        """ Set the Sample Shape Sizes """
        shapeSampleA = self.getCollectionManager().getDesignMatrixA().getSampleShape()
        shapeSampleB = self.getCollectionManager().getDesignMatrixB().getSampleShape()
        
        # Copy Shapes Over
        for i in range(len(shapeSampleA)):
            self._runInfo.getShapeSampleA().append( shapeSampleA[i] )

        for j in range(len(shapeSampleB)):
            self._runInfo.getShapeSampleB().append( shapeSampleB[j] )

        # Return
        return self
            
    def initBatchSizeData(self):
        """ Inititalize Data related to batch Sizes """
        batchSizes = self.getSampleManager().getBatchSizes()
        for i in range(len(batchSizes)):
            self._runInfo.getBatchSizes().append( batchSizes[i] )
        return self

    def initAnalysisFrameParams(self):
        """ Initialize Analysis Frames Paramaters Structure """
        self._frameParams = Structural.AnalysisFramesParameters(
            samplesPerFrame=1024,
            samplesOverlap=768,
            headPad=1024,
            tailPad=2048,
            maxFrames=256,
            window="hanning",
            freqLowHz=0,
            freqHighHz=16000)
        return self

    def initFeatureNamesMatrixA(self):
        """ Initialize the List of all Feature Names for Design Matrix A """
        runQueue = self.getCollectionManager().getMethodQueue()
        featureNames = []
        for item in runQueue:
            # Visit Each Item in the Queue
            if (item == 0):
                # Skip this
                continue
            featureNames += item.featureNames()
        # Attatch to Self + return
        self._runInfo.setFeatureNamesA(featureNames)
        return self

    def initSampleOccuranceData(self):
        """ Initialize the Sample Occurance Data """
        for sample in self.getSampleManager():
            # Iterate through Samples
            self.getOccuranceData().updateExpected(
                sample.getTargetInt(),sample.getTargetStr())

        # Add total number of samples
        self._runInfo.setExpectedNumSamples(len(self.getSampleManager()))
        return self