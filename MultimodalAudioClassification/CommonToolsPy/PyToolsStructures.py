"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifier
Project:        CommonUtilitiesPy
File:           Structural.py
 
Author:         Landon Buell
Date:           April 2022
"""

        #### IMPORTS ####

import os
import numpy as np

import PyToolsIO

        #### FUNCTION DEFINITIONS ####

def getDesignMatrixDataPath(outputPath,batchIndex,pipelineIdentifier):
    """ Get the location of a Design Matrix's Features """
    return os.path.join(outputPath,"batch{0}x-pipeline{1}.bin".format( batchIndex, pipelineIdentifier ) )

def getDesignMatrixLabelsPath(outputPath,batchIndex):
    """ Get the location of a Design Matrix's labels """
    return os.path.join(outputPath,"batch{0}y.bin".format( batchIndex ) )

        #### CLASS DEFINITIONS ####

class FeatureVector:
    """ Class to Hold Feature Data for a single Sample """

    def __init__(self,sampleShape,label=-1):
        """ Constructor for FeatureVector Instance """
        self._sampleShape   = sampleShape
        self._label         = label
        self._data          = np.zeros(shape=sampleShape,dtype=np.float32)

    def __del__(self):
        """ Destructor for FeatureVector Instance """
        self.clearData()

    # Getters and Setters

    def getShape(self):
        """ Get the Shape of this Sample """
        return self._sampleShape

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
            assert(x.shape == self.getShape())
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

    def __getitem___(self,key):
        """ Get the Item at the Index """
        return self._data[key]

    def __setitem__(self,key,value):
        """ Set the Item at the Index """
        value = np.float32(value)   # cast to single-precs
        self._data[key] = value
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

    def getShape(self):
        """ Get Total Shape of Design Matrix """
        shape = [self._numSamples]
        for axisShape in self._sampleShape:
            shape.append(axisShape)
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
        if (numSamples != self._numSamples):
            self._numSamples = numSamples
            self.clearData()
        return self

    def setSampleShape(self,sampleShape):
        """ Set the shape of each sample in the design Matrix """
        if (sampleShape != self._sampleShape):
            self._sampleShape = sampleShape
            self.clearData()
        return self

    def getFeatures(self):
        """ Get Design Matrix as an Array """
        return self._data

    def setFeatures(self,x):
        """ Set Design Matrix is an Array """
        self._numSamples = x.shape[0]
        self._sampleShape = tuple(x.shape[1:])
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
        return np.max(self._tgts)

    # Public Interface

    def __getitem___(self,key):
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
        if (value.getShape() != self.getSampleShape()):
            errMsg = 'Shape mismatch: {0} vs. {1}'.format(value.getShape(),self.getSampleShape())
            raise RuntimeError(errMsg)
        self._tgts[key] = value.getLabel()
        self._data[key] = value.getData()
        return self
  
    def applyMask(self,maskArray):
        """ Apply A Mask Array to samples in the desing Matrix """
        assert(maskArray.shape[0] == self._data.shape[0])
        maskArray = maskArray.astype(np.bool8)
        numSurvivingSamples = np.sum(maskArray)
        newDesignMatrix = DesignMatrix( numSurvivingSamples,self.getSampleShape() )
        sampleIndex = 0

        # Iterate Through Data
        for idx,item in enumerate(maskArray):
            if (item == True):
                newDesignMatrix._data[sampleIndex] = self._data[idx]
                newDesignMatrix._tgts[sampleIndex] = self._tgts[idx]
                sampleIndex += 1

        # Switch Features around
        self.setNumSamples(numSurvivingSamples)
        self._data = newDesignMatrix._data;
        self._tgts = newDesignMatrix._data;
        newDesignMatrix = None
        return self

    def getMaskForNaNsAndInfs(self):
        """ Drop All Rows with NaNs in them """
        sumOfRows = np.sum(self._data,axis=1)
        mask = np.zeros(shape=(self._numSamples,),dtype=np.bool8)
        for idx,item in enumerate(sumOfRows):
            if np.isnan(item) == True:
                continue
            if np.isinf(item) == True:
                continue
            mask[idx] = True
        # When applied, the returned masks will
        # Remove samples w/ NaN or Inf Features
        return mask
        
    def concat(self,otherMatrix):
        """ Concatenate Another Design Matrix to the End of this One (SLOW) """
        if (otherMatrix.getSampleShape() != self.getSampleShape()):
            # Shapes Not Equal
            raise ValueError("Shape Mismatch!")
        totalNumSamples = self.getNumSamples() + otherMatrix.getNumSamples()
        shapeNewMatrix = [totalNumSamples] + [x for x in self.getSampleShape()]
        newFeatureArr = np.empty(shape=shapeNewMatrix,dtype=np.float32)
        # Copy Features to New Array
        sampleIndex = 0
        for smpl in self._data:
            newFeatureArr[sampleIndex] = smpl
            sampleIndex += 1
        for smpl in otherMatrix._data:
            newFeatureArr[sampleIndex] = smpl
            sampleIndex += 1
        # Add to New Design Matrix + Append Target Vector
        self.setFeatures(newFeatureArr)
        self.setLabels(np.append(self._tgts,otherMatrix._tgts))
        return self

    def samplesInClass(self,classIndex):
        """ Create New Design Matrix of Samples that all belong to one class """
        if (classIndex not in self.getUniqueClasses()):
            # Not a Valid Class
            return DesignMatrix(1,self.getSampleShape())
        # Find where targets matches index
        mask = np.where(self._tgts == classIndex)[0]
        newTgts = self._tgts[mask]
        newData = self._data[mask]
        # Create the new Design Matrix, attach values + Return
        result = DesignMatrix(len(mask),self.getSampleShape())
        result.setLabels(newTgts)
        result.setFeatures(newData)
        return result

    def averageOfFeatures(self,mask=None):
        """ Compute the Average of the Design Matrix Along each Feature """
        means = np.mean(self._data,axis=0,dtype=np.float32)
        if (mask is not None):
            means = means[mask]
        return means

    def varianceOfFeatures(self,mask=None):
        """ Compute the Variance of the Design Matrix Along each Feature """
        varis = np.var(self._data,axis=0,dtype=np.float32)
        if (mask is not None):
            varis = varis[mask]
        return varis

    def serialize(self,pathX=None,pathY=None):
        """ Write this design matrix out to a file """   
        writer = DesignMatrix.__DesignMatrixSerializer(self,pathX,pathY)
        success = True
        try:          
            success = writer.call()
        except Exception as err:
            print("\t\tDesignMatrix.serialize() - " + err)
            success = False
        return success

    def clearData(self):
        """ Clear All Entries in this Array """
        self._data = np.zeros(shape=self.getShape(),dtype=np.float32)
        self._tgts = np.zeros(shape=self.getNumSamples(),dtype=np.int16)
        return self

    # Private Interface

    class __DesignMatrixSerializer(PyToolsIO.Serializer):
        """ Class to Serialize a DesignMatrixInstance """
        
        def __init__(self,matrix,pathX=None,pathY=None):
            """ Constructor for DesignMatrixSerializer Instance """
            super().__init__(matrix,None)
            self._pathX =   pathX
            self._pathY =   pathY
            self.__validateOutputs()
            
        def __del__(self):
            """ Destructor for DesignMatrixSerializer Instance """
            super().__del__()

        # Public Interface

        def call(self):
            """ Run the Serializer """
            if (self._pathX is not None):
                self.__writeDataX()
            if (self._pathY is not None):
                self.__writeDataY()
            return self

        # Private Interface

        def __writeDataX(self):
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

        def __writeDataY(self):
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

        def __validateOutputs(self):
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

    class __DesignMatrixDeserializer(PyToolsIO.Deserializer):
        """ Class to Serialize a DesignMatrix Instance """

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
            self.__validateInputPaths()
            self._data.setFeatures( self.__readFeatures() )
            self._data.setLabels( self.__readLabels() )
            return self._data

        # Private Interface

        def __validateInputPaths(self):
            """ Check that Input Directories Exists """
            if (os.path.isfile(self._pathX) == False):
                # Path does not Exist
                FileNotFoundError(self._pathX)
            if (os.path.isfile(self._pathY) == False):
                # Path does not Exist
                FileNotFoundError(self._pathY)
            return True

        def __readFeatures(self):
            """ Read the Feature Data from the File into the Design Matrix """
            shape = self._data.getShape()
            self._inFileStream = open(self._pathX,"rb")
            fileContents = self._inFileStream.read()
            self._inFileStream.close()
            array = np.frombuffer(fileContents,dtype=np.float32)         
            array = array.reshape( shape )         
            return array

        def __readLabels(self):
            """ Read the Feature Data from the File into the Design Matrix """
            self._inFileStream = open(self._pathY,"rb")
            fileContents = self._inFileStream.read()
            self._inFileStream.close()
            array = np.frombuffer(fileContents,dtype=np.int16)               
            return array

    # Static Interface

    @staticmethod
    def deserialize(pathX,pathY,numSamples,shape):
        """ Read a design matrix from a file """
        reader = DesignMatrix.__DesignMatrixDeserializer(
            pathX,pathY,numSamples,shape)
        matrix = reader.call()
        return matrix

    @staticmethod
    def encodeOneHot(targets,numClasses):
        """ Get a One-Hot-Encoded Array of targets """
        numSamples = targets.shape[-1]
        result = np.zeros(shape=(numSamples,numClasses),dtype=np.int16)   
        for i in range(numSamples):
            tgt = targets[i]
            result[i,tgt] = 1
        return result
    
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

class RunInformation:
    """
    Class to Hold and Use all Metadata related to a feature collection Run
    """

    DEFAULT_NUM_PIPELINES = 4

    def __init__(self,inputPaths,outputPath,
                 numSamplesExpected=0,numSamplesRead=0):
        """ Constructor for RunInformation Instance """
        self._pathsInput        = inputPaths
        self._pathOutput        = outputPath

        self._numPipelines      = RunInformation.DEFAULT_NUM_PIPELINES
        self._pipelinesInUse    = [False] * RunInformation.DEFAULT_NUM_PIPELINES

        self._samplesShapes     = [ (0,) ] * RunInformation.DEFAULT_NUM_PIPELINES
        self._batchSizes        = []

        self._numSamplesExpt    = 0
        self._numSamplesRead    = 0

        self._classesInUse      = []


    def __del__(self):
        """ Destructor for RunInformation Instance """
        pass

    # Getters and Setters

    def getRunInfoPath(self) -> str:
        """ Get the Path to the RunInfo Metadata """
        return os.path.join(self._pathOutput,"runInformation.txt")

    def getInputPaths(self) -> set:
        """ Return List of Input Paths """
        return self._pathsInput

    def getOutputPath(self) -> str:
        """ Return Output Path """
        return self._pathOutput

    def getNumPossiblePipelines(self):
        """ Return the possible number of pipelines """
        return RunInformation.DEFAULT_NUM_PIPELINES

    def getPipelinesInUse(self):
        """ Return a List of the index of pipelines in Use """
        return [ii for ii,val in enumerate(self._pipelinesInUse) if val is True]

    def getNumPipelinesInUse(self):
        """ Get the Number of pipelines in Use """
        return sum(self._pipelinesInUse)

    def getSampleShapes(self):
        """ Get the shape of all pipelines """
        return self._samplesShapes

    def getSampleShapeOfPipeline(self,index):
        """ Get the sample Shape f a particular pipeline """
        return self._samplesShapes[index]

    def getBatchSizes(self):
        """ Get the size of all batches """
        return self._batchSizes

    def getSizeOfBatch(self,index):
        """ Get the size of particular index(es) of batches """
        return self._batchSizes[index]

    def getExpectedNumSamples(self):
        """ Get the number of samples expected to process """
        return self._numSamplesExpt

    def getClassesInUse(self):
        """ Get a list of the classes that were passed through the pipelines """
        return self._classesInUse[:]

    def setExpectedNumSamples(self,num):
        """ Set the number of samples expected to process """
        self._numSamplesExpt = num
        return self

    def getActualNumSamples(self):
        """ Get the number of samples actually processed """
        return self._numSamplesRead

    def setActualNumSamples(self,num):
        """ Set the number of samples actually processed """
        self._numSamplesRead = num
        return self

    def getIsPipelineInUse(self,index):
        """ Return T/F If pipeline is in use at index """
        return self._pipelinesInUse[index]

    def setIsPipelineInUse(self,index,value):
        """ Set T/F if pipeline at index is in use """
        self._pipelinesInUse[index] = value
        return self

    def setClassesInUse(self,classesInUse):
        """ Set a List of the Classes that passed through the pipelines """
        self._classesInUse = classesInUse[:]
        return self

    # Public Interface

    def setExpectedNumberOfSamples(self,numSamples):
        """ Set the expected number of samples to be seen by this runtime """
        self._numSamplesExpt = numSamples
        return self

    def registerPipeline(self,pipeline,index):
        """ Register a Pipeline w/ this instance """
        if (pipeline is None):
            self._pipelinesInUse[index] = False
            self._samplesShapes[index] = (0,)
            return self
        # Is Valid Pipeline...
        self._pipelinesInUse[index] = True
        self._samplesShapes[index]  = ( pipeline.getReturnSize(), )
        return self

    def addBatchSize(self,batchSize):
        """ Add Batch Size to list of sizes """
        self._batchSizes.append(batchSize)
        self._numSamplesRead += batchSize
        return self

    def serialize(self,path):
        """ Serialize this Instance """
        success = True
        writer = RunInformation.__RunInformationSerializer(self,path)
        writer.call()
        return success

    def loadSingleBatch(self,batchIndex):
        """ Load Samples from Batch for all pipelines """
        result = [None for x in RunInformation.DEFAULT_NUM_PIPELINES]
        totalNumSamples = self._batchSizes[batchIndex]
        for ii in range(len(result)):
            if (self._pipelinesInUse[ii] == False):
                # This pipeline is not in use
                continue
            # If it is in use...
            pathX = self.__getPathToFile(ii,batchIndex,"X")
            pathY = self.__getPathToFile(ii,batchIndex,"Y")
            sampleShape = self._samplesShapes[ii]
            # Run the deserializer
            result[ii] = DesignMatrix.deserialize(
                pathX,pathY,totalNumSamples,sampleShape)
        # Result Array is Populated - Return now
        return result

    def loadMultipleBatches(self,batchIndices):
        """ Load Samples from multiple batches in all pipelines """
        result = [None for x in RunInformation.DEFAULT_NUM_PIPELINES]
        totalNumSamples = 0
        for batchIndex in batchIndices:
            totalNumSamples += self._batchSizes[batchIndex]
        # For each batch:
        resultSampleCounter = 0
        for ii,batchIndex in enumerate(batchIndices):
            outputMatrixSampleCounter = 0 
            matrices = self.loadSingleBatch(batchIndex)
            self.__populateLargerMatrices(result,matrices,outputMatrixSampleCounter)        
            resultSampleCounter += self._batchSizes[batchIndex]

        return result

    # Private Interface

    def __getPathToFile(self,pipelineIndex,batchIndex,strID):
        """ Get a path to the batch of a particular pipeline """
        fileName = "pipeline{0}Batch{1}{2}.bin".format(
            pipelineIndex,batchIndex,strID)
        return os.path.join(self._outputPath,fileName)

    def __populateLargerMatrices(self,matricesA,matricesB,startIndex):
        """ Populate matrixA w/ contents of matrixB starting at index """
        assert(len(matricesA) == len(matricesB))
        for ii,(a,b) in enumerate(zip(matricesA,matricesB)):
            if ((a is None) or (b is None)):
                # NONE type for matrix
                continue
            indexCounter = startIndex
            for jj in range(len(b.getNumSamples())):
                a[indexCounter] = b[jj]
                indexCounter += 1
        return self

    class RunInfoIOKeys:
        """ Keys used in read/writing RunInformation instance to disk """
        KEY_INPUT_PATHS         = "InputPath"
        KEY_OUTPUT_PATH         = "OutputPath"
        KEY_NUM_PIPELINES       = "NumPipelines"
        KEY_PIPELINES_IN_USE    = "PipelinesInUse"
        KEY_PIPELINE_SHAPE      = "PipelineShape"
        KEY_BATCH_SIZES         = "BatchSizes"
        KEY_EXPECTED_SAMPLES    = "ExpectedSamples"
        KEY_ACTUAL_SAMPLES      = "ActualSamples"
        KEY_CLASSES_IN_USE      = "ClassesInUse"

    class __RunInformationSerializer(PyToolsIO.Serializer):
        """ Class to serialize Run Information Instance"""
 
        def __init__(self,data,path):
            """ Constructor """
            super().__init__(data,path)

        def __del__(self):
            """ Destructor """
            super().__del__()

        # Public Interface

        def call(self):
            """ Write Object to OutputStream """
            self._outFileStream = open(self._outputPath,"w")
            self._writeHeader()

            self.__writeInputOutputPaths()
            self.__writePipelineInUseData()
            self.__writeSampleShapesBatchSizes()
            self.__writeNumSamples()
            self.__writeClassesInUse()
            self._writeFooter()

            # Write to disk + close
            self._outFileStream.writelines(self._buffer)
            self._outFileStream.close()
            return True

        # Private Interface

        def __writeInputOutputPaths(self):
            """ Write Input Paths to File """
            inputPaths = self._data.getInputPaths()
            outputPath = self._data.getOutputPath()

            # write input Paths
            for ii,path in enumerate(inputPaths):
                line = PyToolsIO.Serializer.fmtKeyValPair(
                    RunInformation.RunInfoIOKeys.KEY_INPUT_PATHS + "[{0}]".format(ii),
                    path)
                self.appendLine(line)

            # Write Output Path
            line = PyToolsIO.Serializer.fmtKeyValPair(
                RunInformation.RunInfoIOKeys.KEY_OUTPUT_PATH,
                outputPath)
            self.appendLine(line)

            # All done!
            return self

        def __writePipelineInUseData(self):
            """ Write Data pertaining to what pipeline are in use """
            line = PyToolsIO.Serializer.fmtKeyValPair(
                RunInformation.RunInfoIOKeys.KEY_NUM_PIPELINES,
                self._data.getNumPipelinesInUse() )
            self.appendLine( line )

            # Write out which pipelines are in use
            strPipelineList = PyToolsIO.Serializer.listToString(
                self._data._pipelinesInUse)
            line =  PyToolsIO.Serializer.fmtKeyValPair(
                RunInformation.RunInfoIOKeys.KEY_PIPELINES_IN_USE,
                strPipelineList)
            self.appendLine( line )
            return self
        
        def __writeSampleShapesBatchSizes(self):
            """ Write Sample Shapes and Batch Sizes """
            shapes = self._data.getSampleShapes()

            # Write out shapes
            for ii,shape in enumerate(self._data.getSampleShapes()):
                line = PyToolsIO.Serializer.fmtKeyValPair(
                    RunInformation.RunInfoIOKeys.KEY_PIPELINE_SHAPE + "[{0}]".format(ii),
                    str(shape))
                self.appendLine(line)

            # Write out batch sizes
            strBatchSizes = PyToolsIO.Serializer.listToString(
                self._data.getBatchSizes())
            line = PyToolsIO.Serializer.fmtKeyValPair(
                RunInformation.RunInfoIOKeys.KEY_BATCH_SIZES,
                strBatchSizes) 
            self.appendLine(line)

            return self

        def __writeNumSamples(self):
            """ Write out actual & expected num samples """
            line = PyToolsIO.Serializer.fmtKeyValPair(
                RunInformation.RunInfoIOKeys.KEY_EXPECTED_SAMPLES,
                self._data.getExpectedNumSamples() )
            self.appendLine( line )

            line = PyToolsIO.Serializer.fmtKeyValPair(
                RunInformation.RunInfoIOKeys.KEY_ACTUAL_SAMPLES,
                self._data.getActualNumSamples() )
            self.appendLine( line )
            return self

        def __writeClassesInUse(self):
            """ Write Out the List of Classes Being Used by the pipeline """
            strClassesInUse = PyToolsIO.Serializer.listToString(
                self._data.getClassesInUse())
            line = PyToolsIO.Serializer.fmtKeyValPair(
                RunInformation.RunInfoIOKeys.KEY_CLASSES_IN_USE,
                strClassesInUse)
            self.appendLine(line)
            return self

    class __RunInformationDeserializer(PyToolsIO.Deserializer):
        """ Class to deserialize RunInformation instance """
        
        def __init__(self,path):
            """ Constructor """
            super().__init__(path)

        def __del__(self):
            """ Destructor """
            super().__del__()

        # Public Interface

        def call(self):
            """ Run the deserializer """
            super().call()
            self.__readInputOutputPaths()
            self.__readPipelineInUseData()
            self.__readSampleShapesBatchSizes()
            self.__readNumSamples()
            self.__readClassesInUse()
            return self._data

        # Private Interface

        def __readInputOutputPaths(self):
            """ Write Input Paths to File """
            inputPaths = self._findInBuffer(RunInformation.RunInfoIOKeys.KEY_INPUT_PATHS)
            outputPath = self._findInBuffer(RunInformation.RunInfoIOKeys.KEY_OUTPUT_PATH)[0]

            # Create the Instance
            self._data = RunInformation(inputPaths,outputPath)

            # All done!
            return self

        def __readPipelineInUseData(self):
            """ Write Data pertaining to what pipeline are in use """

            strNumPipelines = self._findInBuffer(RunInformation.RunInfoIOKeys.KEY_NUM_PIPELINES)[0]
            strPipelinesInUse = self._findInBuffer(RunInformation.RunInfoIOKeys.KEY_PIPELINES_IN_USE)[0]
            
            intNumPipelines = int(strNumPipelines[0])
            listPipelinesInUse = PyToolsIO.Deserializer.stringToList(strPipelinesInUse,delimiter=',')
            listPipelinesInUse = [PyToolsIO.Deserializer.stringToBool(x) for x in listPipelinesInUse]

            # Register w/ result instance
            self._data._numPipelines = intNumPipelines
            for ii in range(RunInformation.DEFAULT_NUM_PIPELINES):
                self._data._pipelinesInUse[ii] = listPipelinesInUse[ii]
           
            return self
        
        def __readSampleShapesBatchSizes(self):
            """ Write Sample Shapes and Batch Sizes """
            strSampleShapes = self._findInBuffer(RunInformation.RunInfoIOKeys.KEY_PIPELINE_SHAPE)
            strBatchSizes = self._findInBuffer(RunInformation.RunInfoIOKeys.KEY_BATCH_SIZES)[0]

            # Handle Batch Sizes
            listBatchSizes = PyToolsIO.Deserializer.stringToList(strBatchSizes,delimiter=',',outType=int)
            self._data._batchSizes = listBatchSizes[:]

            # handle Sample Shapes
            self.__readSampleShapesHelper(strSampleShapes)

            return self

        def __readNumSamples(self):
            """ Write out actual & expected num samples """

            expectedSamples = self._findInBuffer(RunInformation.RunInfoIOKeys.KEY_EXPECTED_SAMPLES)[0]
            actualSamples = self._findInBuffer(RunInformation.RunInfoIOKeys.KEY_ACTUAL_SAMPLES)[0]

            self._data._numSamplesExpt = PyToolsIO.Deserializer.stringToInt(expectedSamples)
            self._data._numSamplesRead = PyToolsIO.Deserializer.stringToInt(actualSamples)

            return self

        def __readSampleShapesHelper(self,listOfStrsSampleShape):
            """ Parse out sample shapes from list of strings """
            for ii,strItem in enumerate(listOfStrsSampleShape):
                data = strItem.replace("(","").replace(")","")
                strAxisSizes = data.split(",")
                if (strAxisSizes[-1] == ""):
                    strAxisSizes.pop()
                intAxisSizes = tuple([int(x) for x in strAxisSizes])
                self._data._samplesShapes[ii] = intAxisSizes
            return self

        def __readClassesInUse(self):
            """ Read the List of All Classes Currently In Use """
            strClassesInUse = self._findInBuffer(RunInformation.RunInfoIOKeys.KEY_CLASSES_IN_USE)[0]
            listClassesInUse = PyToolsIO.Deserializer.stringToList(strClassesInUse,delimiter=',')
            listClassesInUse = [int(x) for x in listClassesInUse]
            self._data.setClassesInUse(listClassesInUse)
            return self

    # Static Interface

    @staticmethod
    def deserialize(path):
        """ Deserialize a RunInformation instance from path """
        path = os.path.join(path,"runInfo.txt")
        reader  = RunInformation.__RunInformationDeserializer(path)
        runInfo = reader.call() 
        return runInfo
   
    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))