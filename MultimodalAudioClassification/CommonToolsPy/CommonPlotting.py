"""
Landon Buell
Some Plotty Bois
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import CommonStructures

def scatterPlot(matrix,featureIndex):
    """ Scatter Plot Feature over each class """
    numClasses = matrix.getNumClasses()
    


    return None

def plotSpectrum(x,y):
    """ Plot a 2D Spectrum """
    plt.figure(figsize=(16,12))
    plt.title("")
    plt.xlabel("Amplitude",size=16,weight='bold')
    plt.ylabel("Axis",size=16,weight='bold')
    plt.plot(x,y,color='blue')
    plt.grid()
    plt.show()
    return None

def plotSpectrogram(matrix,time=None,freq=None,title=""):
    """ Plot a 2D Spectrum """
    plt.figure(figsize=(16,8))
    plt.title(title,size=20,weight='bold')
    plt.xlabel("Frequency",size=16,weight='bold')
    plt.ylabel("Time",size=16,weight='bold')
    if (type(matrix) == CommonStructures.FeatureVector):
        matrix = matrix.getData()
    plt.imshow(X=matrix,interpolation='none')
    plt.grid()
    plt.show()
    return None

def plotBoxAndWhisker(boxPlotData,title,xlabels,save=None):
    """ Plot Box-And-Whisker Diagram """
    plt.figure(figsize=(20,8))
    plt.title(title,size=32,weight='bold')
    
    boxPlots = [x._data for x in boxPlotData]
    plt.boxplot(boxPlots,showfliers=False)
    
    plt.grid()
    plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.2)
    plt.yticks(weight='bold',rotation=80)
    plt.xticks(ticks=range(1,len(boxPlots)+1,1),labels=xlabels,
               rotation=80,weight='bold')
    plt.hlines(0,0.5,len(xlabels)+0.5,color='black')

    if (save is not None):
        save = str(save)
        plt.savefig(save)

    plt.close()
    return None

class BoxPlotGenerator:
    """ Generate Box Plots for Features """

    def __init__(self,matrix,runInfo,classData):
        """ Constructor for BoxPlotGenerator """
        self._matrix = matrix
        self._runInfo = runInfo
        self._classData = classData
        self._boxData = []

    def __del__(self):
        """ Destructor for BoxPlotGenerator """
        pass

    def call(self,outputPath):
        """ Generate the Box Plots """
        os.makedirs(outputPath,exist_ok=True)
        numFeatures = self._matrix.getNumFeatures()
        uniqueClasses = self._matrix.getUniqueClasses()
        classNames = [self._classData.getNameFromInt(x) for x in uniqueClasses]

        # Iterate through each feature
        for i in range(numFeatures):
            featureData = self._matrix._data[:,i]
            featureName = self._runInfo._featureNamesA[i]
            self._boxData.clear()

            # Iterate through each class
            for j in uniqueClasses:
                classRows = np.where(self._matrix._tgts == j)
                classData = featureData[classRows]
                className = self._classData.getNameFromInt(j)
                self._boxData.append(BoxPlotGenerator.BoxPlotData(classData,className))

            # Plot each Class in this Features
            savePath = os.path.join(outputPath,featureName + ".png")
            plotBoxAndWhisker(self._boxData,featureName,classNames,savePath)

        return self


    class BoxPlotData:
        """ Store Data for a single Box Plot """
        
        def __init__(self,data,label):
            """ Constructor for BoxPlotData Instance """
            self._data = data
            self._label = label

        def __del__(self):
            """ Destructor for BoxPlotData Instance """
            pass

        def makeData(self):
            """ Make the Boxplot Data """
            dataMin = np.min(self._data)
            dataMax = np.max(self._data)
            quants = np.quantile(self._data,[0.25,0.5,0.75])
            return np.array([dataMin,quants,dataMax]).flatten()