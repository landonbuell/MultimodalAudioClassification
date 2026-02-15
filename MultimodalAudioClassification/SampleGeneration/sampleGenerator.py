"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    SampleGeneration
    File:       sampleGenerator.py
    Classes:    SampleGenerator

    Author:     Landon Buell
    Date:       February 2025
"""

        #### IMPORTS ####

import sampleGeneratorTypes
import sampleGeneratorCallbacks

        #### CLASS DEFINITIONS ####

class SampleGenerator:
    """ 
        Base Class for All Sample Generators
    """

    __COUNT = 0

    def __init__(self,
                 params: sampleGeneratorTypes.SampleGenerationParameters,
                 callback: sampleGeneratorCallbacks.SampleGenerationCallback,
                 drawLimit: int = 1024,
                 classIndex: int = -1,
                 className: str = ""):
        """ Constructor """
        self._params    = params
        self._callback  = callback
        self._drawLimit = drawLimit
        self._drawCount = 0
        self._classIndex    = classIndex
        self._className     = className

        if (self._className == ""):
            # No Name provided
            self._className = "SampleGenerator{0}".format(SampleGenerator.__COUNT)
            SampleGenerator.__COUNT += 1

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def drawCount(self) -> int:
        """ Return the number of  draws """
        return self._drawCount

    def drawsRemaining(self) -> int:
        """ Return the number of draws remaining """
        return self._drawLimit - self._drawCount

    def drawLimit(self) -> int:
        """ Return the limit on the number of draws """
        return self._drawLimit

    def isEmpty(self) -> bool:
        """ Return if the drawLimit has been reached """
        return (self._drawCount >= self._drawLimit)

    def getClassIndex(self) -> int:
        """ Return the class that this generator draws for """
        return self._classIndex

    def getClassName(self) -> str:
        """ Return the class name for this generator """
        return self._className
     
    def getSampleRate(self) -> float:
        """ Return the sample rate for the generated samples """
        return self._params.sampleRate

    # Public Interface

    def drawNext(self) -> sampleGeneratorTypes.GeneratedSample:
        """ Draw a sample """
        if (self.isEmpty() == True):
            msg = "Draw limit reached on {0}".format(self)
            raise RuntimeError(msg)
        generatedSample = self.__generateNextSample()  
        self._drawCount += 1
        return generatedSample

    def resetDrawCount(self) -> None:
        """ Reset the internal draw counter """
        self._drawCount = 0
        return None

    # Private

    def __generateNextSample(self) -> sampleGeneratorTypes.GeneratedSample:
        """ Invoke the callback to generate a sample """
        generatedSample = self._callback(self._params) 
        generatedSample.classInt = self._classIndex
        generatedSample.sampleRate = self._params.sampleRate
        return generatedSample

    # Dunder

    def __str__(self) -> str:
        """ Cast to string """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0}: ({1}/{2})".format(str(self),self._drawCount,self._capacity)


