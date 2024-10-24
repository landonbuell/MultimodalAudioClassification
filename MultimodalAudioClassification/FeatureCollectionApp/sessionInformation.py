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
            self.index      = classInt
            self.name       = classStr
            self.expectedCount = 0
            self.processedCount = 0
            self.exportedCount = 0

        def __del__(self):
            """ Destructor """
            pass

        def __repr__(self) -> str:
            """ Return instance as string """
            return "{0}:{1} - ({2},{3},{4})".format(
                self.index,self.name,
                self.expectedCount,
                self.processedCount,
                self.exportedCount)

        def __str__(self) -> str:
            """ Return instance as string """
            return "{0:<8}{1:<24}{2:<8}{3:<8}{4:<8}".format(
                self.index,self.name,
                self.expectedCount,
                self.processedCount,
                self.exportedCount)

    def __init__(self):
        """ Constructor """
        self._classMap  = dict()

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

    # Public Interface

    def registerClass(self, classIndex: int, className: str) -> None:
        """ Register a new class to the database """
        if (self.hasClassIndex(classIndex) == False):
            self._classMap[classIndex] = ClassInfoDatabase.ClassInfo(classIndex,className)
        return None

    def incrementExpectedCount(self, classIndex: int) -> None:
        """ Increment the number of times we expect to process this class """
        self._classMap[classIndex].expectedCount += 1
        return None

    def incrementProcessedCount(self, classIndex: int) -> None:
        """ Increment the number of times we've processed this class """
        self._classMap[classIndex].processedCount += 1
        return None

    def incrementExportedCount(self, classIndex: int) -> None:
        """ Increment the number of times we've exported this class's features """
        self._classMap[classIndex].exportedCount += 1
        return None

