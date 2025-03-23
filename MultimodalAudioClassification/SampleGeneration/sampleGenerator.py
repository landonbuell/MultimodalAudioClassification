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

    def __init__(self,
                 params: sampleGeneratorTypes.SampleGenerationParameters,
                 callback: sampleGeneratorCallbacks.SampleGenerationCallback,
                 drawLimit: int = 1024):
        """ Constructor """
        self._params = params
        self._callback = callback
        self._drawCount = 0

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def drawCount(self) -> int:
        """ Return the number of  draws """
        return self._drawCount

    def drawLimit(self) -> int:
        """ Return the limit on the number of draws """
        return self._config.drawLimit

    def isEmpty(self) -> bool:
        """ Return if the drawLimit has been reached """
        return (self._drawCount >= self.drawLimit())

    # Public Interface

    def drawNext(self) -> sampleGeneratorTypes.GeneratedSample:
        """ Draw a sample """
        if (self.isEmpty() == True):
            msg = "Draw limit reached on {0}".format(self)
            raise RuntimeError(msg)
        generatedSample = self.__generateNextSample()  
        return generatedSample

    def resetDrawCount(self) -> None:
        """ Reset the internal draw counter """
        self._drawCount = 0
        return None

    # Private

    def __generateNextSample(self) -> sampleGeneratorTypes.GeneratedSample:
        """ Invoke the callback to generate a sample """
        generatedSample = self._callback(self._params) 
        return generatedSample

    # Dunder

    def __str__(self) -> str:
        """ Cast to string """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0}: ({1}/{2})".format(str(self),self._drawCount,self._capacity)


