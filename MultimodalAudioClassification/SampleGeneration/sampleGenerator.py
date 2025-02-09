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

        #### CLASS DEFINITIONS ####

class SampleGenerator:
    """ 
        Base Class for All Sample Generators
    """

    def __init__(self,
                 className: str,
                 classIndex: int,
                 drawLimit: int,
                 sampleGeneratorCallback: function,
                 waveformParams: sampleGeneratorTypes.SampleGenerationParameters,):
        """ Constructor """
        self._name      = className
        self._index     = classIndex
        self._drawCount = 0
        self._capacity  = drawLimit

        self._generatorCallback  = sampleGeneratorCallback
        self._waveformParams     = waveformParams

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getClassName(self) -> str:
        """ Return the name of the class that this generator creates """
        return self._name

    def getClassIndex(self) -> int:
        """ Return the index of the class that this generator creates """
        return self._index

    def drawCount(self) -> int:
        """ Return the number of  draws """
        return self._drawCount

    def drawLimit(self) -> int:
        """ Return the limit on the number of draws """
        return self._capacity

    def isEmpty(self) -> bool:
        """ Return if the drawLimit has been reached """
        return (self._drawCount >= self._capacity)

    def params(self) -> sampleGeneratorTypes.SampleGenerationParameters:
        """ Return sample generation parameters """
        return self._waveformParams

    # Public Interface

    def drawNext(self) -> np.ndarray:
        """ Draw a sample """
        if (self.isEmpty() == True):
            msg = "Draw limit reached on {0}".format(self)
            raise RuntimeError(msg)
        sample = self.__generateSample()
        return sample

    def resetDrawCount(self) -> None:
        """ Reset the internal draw counter """
        self._drawCount = 0
        return None

    # Private

    def __generateSample(self) -> np.ndarray:
        """ Invoke the callback to generate a sample """
        return self._generatorCallback.__call__(self._waveformParams)

    # Dunder

    def __str__(self) -> str:
        """ Cast to string """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))

    def __repr__(self) -> str:
        """ Debug representation """
        return "{0}: ({1}/{2})".format(str(self),self._drawCount,self._capacity)


