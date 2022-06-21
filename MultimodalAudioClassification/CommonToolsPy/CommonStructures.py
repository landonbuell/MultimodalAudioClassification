"""
Repository:     MultimodalAudioClassification
Solution:       MultimodalAudioClassification
Project:        CommonToolsPy
File:           CommonStructures.py
 
Author:         Landon Buell
Date:           June 2022
"""

        #### IMPORTS ####

import os
import sys

import numpy as np

import CommonIO


        #### CLASS DEFINITIONS ####

class FeatureVector:
    """ Class to Hold Feature Data for a single Sample """

    def __init__(self,sampleShape,label=-1):
        """ Constructor for FeatureVector Instance """
        self._label         = label
        self._data          = np.zeros(shape=sampleShape,dtype=np.float32)

    def __del__(self):
        """ Destructor for FeatureVector Instance """
        self.clearData()

    # Getters and Setters

    def getShape(self):
        """ Get the Shape of this Sample """
        return self._data.shape

    def getLabel(self):
        """ Get the Target Label """
        return self._label

    def setLabel(self,x):
        """ Set the Target Label """
        self._label = x
        return self

    def getData(self):
        """ Get the underlying Array """
        return self._data

    def setData(self,x,enforceShape=True):
        """ Set the Underlying Array, optionally chanign shape """
        if (enforceShape == True):
            assert(list(x.shape) == self.getShape())
            self._data = x
        else:
            self._sampleShape = x.shape
            self._data = x
        return self

    # Public Interface

    def clearData(self):
        """ Clear All Entries in this Array """
        self._label         = -1
        self._data          = np.zeros(shape=self._sampleShape,dtype=np.float32)
        return self

    # Magic Method

    def __str__(self):
        """ String Representation of Instance """
        return str(self.__class__) + " w/ shape: " + str(self.getShape())

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

    def __iter__(self):
        """ Forward-Iterator through Design Matrix """
        for i in range(self._data.shape[0]):
            yield self._data[i]

    def __getitem___(self,key):
        """ Get the Item at the Index """
        return self._data[key]

    def __setitem__(self,key,value):
        """ Set the Item at the Index """
        self._data[key] = np.float32(value)
        return self

class DesignMatrix:
    """ Class To hold Design Matrix """

    def __init__(self,numSamples: int,sampleShape: tuple):
        """ Constructor for DesignMatrix Instance """
        self._numSamples    = numSamples 
        self._sampleShape   = sampleShape
        self._data          = np.zeros(shape=self.getShape(),dtype=np.float32)
        self._tgts          = np.zeros(shape=numSamples,dtype=np.int16)

    def __del__(self):
        """ Destructor for DesignMatrix Instance """
        self.clearData()

    # Getters and Setters

    def getMatrixShape(self):
        """ Get Total Shape of Design Matrix """
        shape = [self._numSamples] + [x for x in self._sampleShape.shape]
        return tuple(shape)

    def getSampleShape(self):
        """ Get the Shape of Each Sample in the Design Matrix """
        return self._sampleShape

    def getNumFeatures(self):
        """ Get the Total Number of Features for each sample """
        numFeatures = 1
        for axisSize in self._sampleShape:
            numFeatures *= axisSize
        return numFeatures

    def getNumSamples(self) -> int:
        """ Get the Number of Samples in the Design Matrix """
        return self._numSamples

    def setNumSamples(self,numSamples):
        """ Set the Number of Samples in the Design Matrix """
        self._numSamples = numSamples
        self.clearData()
        return self

    def getFeatures(self):
        """ Get Design Matrix as an Array """
        return self._data

    def setFeatures(self,x):
        """ Set Design Matrix is an Array """
        self._numSamples = x.shape[0]
        self._sampleShape = tuple(x.shape[1:])
        self._data = None
        self._data = x
        return self

    def getLabels(self):
        """ Get the Labels as an Array """
        return self._tgts

    def setLabels(self,x):
        """ Set the Labels as an Array """
        self._tgts = x
        return self

    def getUniqueClasses(self):
        """ Get An Array of the unique classes """
        return np.unique(self._tgts)

    def getNumClasses(self):
        """ Get the Number of classes in the data set """
        return np.unique(self._tgts).shape[-1]

    # public Interface

    def serialize(self,pathX=None,pathY=None):
        """ Write this design matrix out to a file """   
        writer = DesignMatrix.DesignMatrixSerializer(self,pathX,pathY)
        success = True
        try:          
            writer.call()
        except Exception as err:
            print("\t\tDesignMatrix.serialize()" + err)
            success = False
        return success

    @staticmethod
    def deserialize(pathX,pathY,numSamples,shape):
        """ Read a design matrix from a file """
        reader = DesignMatrix.DesignMatrixDeserializer(
            pathX,pathY,numSamples,shape)
        matrix = reader.call()
        return matrix

    def clearData(self):
        """ Clear All Entries in this Array """
        self._data = np.zeros(shape=self.getShape(),dtype=np.float32)
        self._tgts = np.zeros(shape=self.getNumSamples(),dtype=np.int16)
        return self

    @staticmethod
    def encodeOneHot(targets,numClasses):
        """ Get a One-Hot-Encoded Array of targets """
        numSamples = targets.shape[-1]
        result = np.zeros(shape=(numSamples,numClasses),dtype=np.int16)   
        for i in range(numSamples):
            tgt = targets[i]
            result[i,tgt] = 1
        return result

    # Private Interface

    class DesignMatrixSerializer(CommonIO.Serializer):
        """ Class to Serialize a DesignMatrixInstance """
        
        def __init__(self,matrix,pathX=None,pathY=None):
            """ Constructor for DesignMatrixSerializer Instance """
            super().__init__(matrix,None)
            self._pathX =   pathX
            self._pathY =   pathY
            
            
        def __del__(self):
            """ Destructor for DesignMatrixSerializer Instance """
            super().__del__()

        def call(self):
            """ Run the Serializer """
            self.validateOutputs()
            if (self._pathX is not None):
                self.writeDataX()
            if (self._pathY is not None):
                self.writeDataY()
            return self

        def writeDataX(self):
            """ Write the Design Matrix Data """
            numSamples = self._data.getNumSamples()
            X = self._data.getFeatures()
            self._outFileStream = open(self._pathX,"wb")
            for i in range(numSamples):
                row = X[i].flatten().tobytes()
                self._outFileStream.write( row )
            # Close + Return
            self._outFileStream.close()
            return self

        def writeDataY(self):
            """ Write the Design Matrix Labels """
            numSamples = self._data.getNumSamples()
            Y = self._data.getLabels()
            self._outFileStream = open(self._pathY,"wb")
            for i in range(numSamples):
                row = Y[i].flatten().tobytes()
                self._outFileStream.write( row )
            # Close + Return
            self._outFileStream.close()
            return self

        def validateOutputs(self):
            """ Validate that Both Output Paths Make Sense """
            if (self._pathX is None and self._pathY is None):
                # Both Cannot be none - Nothing will be written
                errMsg = "Both X and Y export paths cannot be None"
                raise ValueError(errMsg)
            elif (self._pathX == self._pathY):
                # Both cannot be the same - will overwrite each other
                errMsg = "X and Y paths cannot be indentical"
                raise ValueError(errMsg)
            else:
                return self

    class DesignMatrixDeserializer(CommonIO.Deserializer):
        """ Class to Deserialize a DesignMatrix Instance """

        def __init__(self,pathX,pathY,numSamples,sampleShape):
            """ Constructor for DesignMatrixSerializer Instance """
            super().__init__("-1")
            self._pathX = pathX
            self._pathY = pathY
            self._data = DesignMatrix(numSamples,sampleShape)

        def __del__(self):
            """ Destructor for DesignMatrixSerializer Instance """
            super().__del__()

        def call(self):
            """ Run the Deserializer """
            self.validateInputPaths()
            self._data.setFeatures( self.readFeatures() )
            self._data.setLabels( self.readLabels() )
            return self._data

        # Private Interface

        def validateInputPaths(self):
            """ Check that Input Directories Exists """
            if (os.path.isfile(self._pathX) == False):
                # Path does not Exist
                FileNotFoundError(self._pathX)
            if (os.path.isfile(self._pathY) == False):
                # Path does not Exist
                FileNotFoundError(self._pathY)
            return True

        def readFeatures(self):
            """ Read the Feature Data from the File into the Design Matrix """
            shape = self._data.getShape()
            self._inFileStream = open(self._pathX,"rb")
            fileContents = self._inFileStream.read()
            self._inFileStream.close()
            array = np.frombuffer(fileContents,dtype=np.float32)         
            array = array.reshape( shape )         
            return array

        def readLabels(self):
            """ Read the Feature Data from the File into the Design Matrix """
            self._inFileStream = open(self._pathY,"rb")
            fileContents = self._inFileStream.read()
            self._inFileStream.close()
            array = np.frombuffer(fileContents,dtype=np.int16)               
            return array
 
    # Magic Methods 

    def __str__(self):
        """ String Representation of Instance """
        return str(self.__class__) + " w/ shape: " + str(self.getShape())

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

    def __iter__(self):
        """ Forward-Iterator through Design Matrix """
        for i in range(self._data.shape[0]):
            yield self._data[i]

    def __getitem__(self,key):
        """ Get the Item at the Index """
        if (key < 0 or key >= self._numSamples):
            errMsg = "key index is out of range for " + self.__repr__
            raise IndexError(errMsg)
        # Make a Feature Vector + Return it
        featureVector = FeatureVector(self._sampleShape,self._tgts[key])
        featureVector.setData(self._data[key])
        return featureVector

    def __setitem__(self,key,value):
        """ Set the Item at the Index """
        assert(value.getShape() == self._sampleShape)
        self._tgts[key] = value.getLabel()
        self._data[key] = value.getData()
        return self
a
class RunInformation:
    """
    Class to Hold and Use all Metadata related to a feature collection Run
    """

    def __init__(self,inputPaths,outputPath,
                 numSamplesExpected=0,numSamplesRead=0):
        """ Constructor for RunInformation Instance """
        self._pathsInput        = inputPaths
        self._pathOutput        = outputPath

        self._batchSizes        = []
        self._pipelineInfo      = []


    def __del__(self):
        """ Destructor for RunInformation Instance """
        pass

    # Getters and Setters

    def getRunInfoPath(self):
        """ Get the Path to the RunInfo Metadata """
        return os.path.join(self._pathOutput,"runInfo.txt")

    def getInputPaths(self) -> set:
        """ Return List of Input Paths """
        return self._pathsInput

    def getOutputPath(self) -> str:
        """ Return Output Path """
        return self._pathOutput

    def getNumSamplesRead(self):
        """ Get the number of samples actually processed """
        return self._numSamplesRead

    def getBatchSizes(self):
        """ Get a list with the size of each batch """
        return self._batchSizes

    def getNumBatches(self):
        """ Get the Number of Batches in the run """
        return len(self._batchSizes)
    
    # Public Interface 

    def serialize(self,path):
        """ Serialize this Instance to specified Path """
        writer = RunInformation.RunInformationSerializer(self,path)
        success = False
        try:
            success = writer.call()
        except Exception as expt:
            print(expt)
        return success

    @staticmethod
    def deserialize(path):
        """ Deserialize this instance from specified path """
        runInfoPath = os.path.join(path,"runInformation.txt")
        if (os.path.isfile(runInfoPath) == False):
            # RunInfo File
            errMsg = "ERROR: run information file not found at '{0}' ".format(runInfoPath)
            FileNotFoundError(errMsg)
        reader = RunInformation.RunInformationDeserializer(runInfoPath)
        runInfo = reader.call()
        return runInfo


    # Private Interface

    class RunInformationSerializer(CommonIO.Serializer):
        """ Class to Serialize Run Information to a Local Path """

        def __init__(self,runInfo,path):
            """ Constructor for RunInformationSerializer Instance """
            super().__init__(runInfo,path)

        def __del__(self):
            """ Destructor for DesignMatrixSerializer Instance """
            super().__del__()

        def call(self):
            """ Serialize the RunInfo Instance """
            self.writeStandardInfo()
            self.writeFeatureNames()
            return self

        def writeStandardInfo(self):
            """ Serialize the RunInfo Instance """          

            self._outFileStream = open(self._outputPath,"w")
            self.writeHeader()

            

            # Close + Return
            self.writeFooter()
            self._outFileStream.close()
            return True

        def writeFeatureNames(self):
            """ Write out the Name of Features """
            outpath = os.path.split(self._outputPath)
            self._outFileStream = open(os.path.join(outpath[0],"featureNames.txt"),"w")
            self.writeHeader()

            # Write Each Feature Name
            for idx,name in enumerate(self._data.getFeatureNamesA()):
                outLine = self._outFmtStr(str(idx),name)
                self._outFileStream.write(outLine)

            self.writeFooter()
            self._outFileStream.close()
            return True

    class RunInformationDeserializer(CommonIO.Deserializer):
        """ Class to Deserialize Run Information from a Local Path """

        def __init__(self,path):
            """ Constructor for RunInformationSerializer Instance """
            super().__init__(path)
            self._inFileContents = None

        def __del__(self):
            """ Destructor for DesignMatrixSerializer Instance """
            super().__del__()
            self._inFileContents = None

        def call(self):
            """ Serialize the RunInfo Instance """    
            self.readStandardInfo()
            self.readFeatureNames()
            return self._data

        def readStandardInfo(self):
            """ Read All Standard Run Information + Populate Self """
            self._inFileStream = open(self._inputPath,"r")
            self._inFileContents = self._inFileStream.readlines()
            
            # Find all of the Necessary parts
            self._data = self.parseAllFeilds()

            # Close + De Allocate
            self._inFileStream.close()
            self._inFileContents = None
            return self

        def readFeatureNames(self):
            """ Read in the name of each Feature """
            inPath = os.path.split(self._inputPath)
            self._inFileStream = open(os.path.join(inPath[0],"featureNames.txt"),"r")
            self._inFileContents = self._inFileStream.readlines()

            # Fin all of the Neccessary Parts
            self.loadFeatureNamesA()

            # Close + De Allocate
            self._inFileStream.close()
            self._inFileContents = None
            return self

        # Private Interface

        def parseAllFeilds(self):
            """ Find all of the Feilds to Create the RunInformation Instance """
            
            # Parse the feilds from the RunInfo File
            pathsInput      = self.findAndParseStrs("InputPath")
            pathOutput      = self.findAndParseStrs("OutputPath")[-1]
            samplesExpected = self.findAndParseInts("TotalNumSamples")[-1]
            samplesActual   = self.findAndParseInts("ProcessedSamples")[-1]
            shapeSampleA    = self.findAndParseInts("ShapeSampleA")
            shapeSampleB    = self.findAndParseInts("ShapeSampleB")
            #featureNamesA   = self.findAndParseStrs("FeatureNamesA")[-1].split(",")
            #featureNamesB   = self.findAndParseStrs("FeatureNamesB")
            batchSizes      = self.findAndParseInts("BatchSizes")
            
            # Assign the Feilds to the instance
            runInfo = RunInformation(pathsInput,pathOutput)
            runInfo.setExpectedNumSamples(samplesExpected)
            runInfo.setActualNumSamples(samplesActual)
            runInfo.setShapeSampleA(shapeSampleA)
            runInfo.setShapeSampleB(shapeSampleB)
            #runInfo.setFeatureNamesA(featureNamesA)
            #runInfo.setFeatureNamesB(featureNamesB)
            runInfo.setBatchSizes(batchSizes)
            return runInfo

        def loadFeatureNamesA(self):
            """ Find all of the feature Names """
            numFeatures = self._data.getShapeSampleA()[0]
            numLines = len(self._inFileContents)
            nameList = ["p" + str(x) for x in range(numFeatures)]
            
            for line in self._inFileContents:

                if (line.startswith("-") or line.startswith("<")):
                    # Check for header or footer
                    continue

                try:
                    # Tokenize + Assign by index
                    tokens = line.split()
                    index = int(tokens[0])
                    name = tokens[1]
                    nameList[index] = name

                except Exception as err:
                    # Note a problem?
                    print(err)
                    continue               
            # attach to name list to the internal struct
            self._data.setFeatureNamesA(nameList)
            return self


        def findAndParseStrs(self,keyword):
            """ Find All words with token and return as list of Strings"""
            result = []
            for line in self._inFileContents:
                tokens = line.split()
                if tokens[0].startswith(keyword):
                    result.append(tokens[-1].strip())
            return result

        def findAndParseInts(self,keyword):
            """ Find All words with token and return as list of Strings"""
            result = self.findAndParseStrs(keyword)
            result = result[0].split(',')
            result = ["".join(ch for ch in x if ch.isdigit()) for x in result]
            result = [int(x) for x in result if x != '']
            return result
        
    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))
