"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureCollection
    File:       sessionInformation.py
    Classes:    RunInfo,
                ClassInfo

    Author:     Landon Buell
    Date:       April 2024
"""

        #### IMPORTS ####

        #### CLASS DEFINITIONS ####

class RunInfo:
    """ Store general run-session information """

    def __init__(self):
        """ Constructor """
        pass

    def __del__(self):
        """ Destructor """
        pass

class ClassInfoDatabase:
    """ Store info about each class + occurence """

    class ClassInfo:
        """ Stores info about a single Class """

        def __init__(self,
                     classInt: int,
                     classStr: str):
            """ Constructor """
            self.index          = classInt
            self.name           = classStr
            self.expectedCount  = 0
            self.processedCount = 0
            self.exportedCount  = 0

        def __del__(self):
            """ Destructor """
            pass

        @staticmethod
        def formatString(col0,
                         col1,
                         col2,
                         col3,
                         col4) -> str:
            """ Return a formated string """
            return "{0:<16}{1:<32}{2:<16}{3:<16}{4:<16}".format(
                col0,col1,col2,col3,col4)

        def __repr__(self) -> str:
            """ Return instance as string """
            return "{0}:{1} - ({2},{3},{4})".format(
                self.index,self.name,
                self.expectedCount,
                self.processedCount,
                self.exportedCount)

        def __str__(self) -> str:
            """ Return instance as string """
            return ClassInfoDatabase.ClassInfo.formatString(
                self.index,self.name,
                self.expectedCount,
                self.processedCount,
                self.exportedCount)

    def __init__(self):
        """ Constructor """
        self._classMap  = dict() # int -> ClassInfoDatabase.ClassInfo

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getClassName(self, classIndex: int) -> str:
        """ Return the name of the class based on the integer index """
        return self._classMap[classIndex].name

    def getClassIndex(self, className: str) -> int:
        """ Return the index of the class based on it's name """
        for (key,val) in self._classMap.items():
            if (val == className):
                return key
        return -1

    def getExpectedCounts(self, classIndex: int) -> int:
        """ Return the number of times we expected to process this class """
        return self._classMap[classIndex].expectedCount

    def getProcessedCount(self, classIndex: int) -> int:
        """ Return the number of times we actually processed this class """
        return self._classMap[classIndex].processedCount

    def getExportedCount(self, classIndex: int) -> int:
        """ Return the number of times we actually exported this class's features """
        return self._classMap[classIndex].exportedCount

    def hasClassIndex(self, classIndex: int) -> bool:
        """ Return T/F if data for this class index already exists """
        return (classIndex in self._classMap.keys())

    def hasClassName(self, className: str) -> bool:
        """ Return T/F if data for this class name alread exists """
        index = self.getClassIndex(className)
        return (index != -1)

    def getTotals(self) -> list:
        """ Return a list of [totalNumExpected, totalNumProcessed, totalNumExported] """
        result = [0,0,0]
        for classIndex,classData in self._classMap.items():
            result[0] += classData.expectedCount
            result[1] += classData.processedCount
            result[2] += classData.exportedCount
        return result


    # Public Interface

    def registerClass(self, classIndex: int, className: str) -> None:
        """ Register a new class to the database """
        if (self.hasClassIndex(classIndex) == False):
            self._classMap[classIndex] = ClassInfoDatabase.ClassInfo(classIndex,className)
        return None

    def incrementExpectedCount(self, 
                               classIndex: int, 
                               count = 1) -> None:
        """ Increment the number of times we expect to process this class """
        self._classMap[classIndex].expectedCount += count
        return None

    def incrementProcessedCount(self,
                                classIndex: int,
                                count = 1) -> None:
        """ Increment the number of times we've processed this class """
        self._classMap[classIndex].processedCount += count
        return None

    def incrementExportedCount(self,
                               classIndex: int,
                               count = 1) -> None:
        """ Increment the number of times we've exported this class's features """
        self._classMap[classIndex].exportedCount += count
        return None

    def exportToFile(self,
                     outputPath: str) -> bool:
        """ Export this instance to the provided text file """
        with open(outputPath,"w") as outputStream:
            # Write Header
            header = ClassInfoDatabase.ClassInfo.formatString(
                col0="CLASS_INDEX",
                col1="CLASS_NAME",
                col2="NUM_EXPECTED",
                col3="NUM_PROCESSED",
                col4="NUM_EXPORTED")
            outputStream.write(header + "\n")
            # Write body
            for (key,val) in self._classMap.items():
                outputStream.write(str(val) + "\n")
        return None

    def readFromFile(self,inputPath: str) -> None:
        """ Import this instance from the provided path """
        with open(inputPath,"r") as inputStream:
            for ii,line in enumerate(inputStream):
                if (ii == 0):
                    # skip the header
                    continue
                # Tokeninze the line
                lineTokens = line.strip().split()
                classIndex  = int(lineTokens[0])
                self.registerClass(classIndex,lineTokens[1])
                self.incrementExpectedCount(classIndex,int(lineTokens[2]))
                self.incrementProcessedCount(classIndex,int(lineTokens[3]))
                self.incrementExportedCount(classIndex,int(lineTokens[4]))
            # Done reading file
        return None