"""
    Repo:       MultiModalAudioClassification
    Solution:   MultiModalAudioClassification
    Project:    FeautureEngineering
    File:       visuals.py
    Classes:    NONE

    Author:     Landon Buell
    Date:       Jan 2025
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt

        #### FUNCTIONS DEFINITIONS ####

def plotHeatmap(array: np.ndarray) -> None:
    """ Show a spectrogram """
    plt.figure(figsize=(16,8))
    plt.pcolormesh(array,cmap="jet",edgecolor=None)
    plt.tight_layout()
    plt.show()
    return None

def plotSpectrogram(spectrogram: np.ndarray) -> None:
    """ Show a spectrogram """   
    toShow = spectrogram
    if (spectrogram.ndim == 3):
        realSq = spectrogram[:,:,0]**2
        imagSq = spectrogram[:,:,1]**2
        toShow = (realSq * imagSq)
    toShow = np.log(toShow)
    plotHeatmap(toShow)
    return None

def plotSimpleLine(array : np.ndarray,
                 xAxis=None,
                 xlabel="",
                 ylabel="",
                 title="") -> None:
    """ Simple1D Plot """
    if (xAxis is None):
        xAxis = np.arange(array.shape[0])
    plt.figure(figsize=(16,8))
    if (title != ""):
        plt.title(title,size=40,weight='bold')
    plt.plot(xAxis,array,marker='o',markersize=16)
    plt.xlabel(xlabel,size=32,weight='bold')
    plt.ylabel(ylabel,size=32,weight='bold')
    plt.grid()
    plt.tight_layout()
    plt.show()


class FeaturesOfAxes2D:
    """ Tools to explore the qualities of rows & cols of 2D arrays """

    @staticmethod
    def plotMeanOfRows(
        array2D: np.ndarray) -> None:
        """ Take a 2D array and plot the AVERAGE of each ROW """
        data = np.mean(array2D,axis=0)
        plotSimpleLine(data,None,"Row Index","Mean Value")
        return None

    @staticmethod
    def plotMeanOfCols(
        array2D: np.ndarray) -> None:
        """ Take a 2D array and plot the AVERAGE of each COLUMN """
        data = np.mean(array2D,axis=1)
        plotSimpleLine(data,None,"Col Index","Mean Value")
        return None

    @staticmethod
    def plotVarianceOfRows(
        array2D: np.ndarray) -> None:
        """ Take a 2D array and plot the VARIANCE of each ROW """
        data = np.var(array2D,axis=0)
        plotSimpleLine(data,None,"Row Index","Variance Value")
        return None

    @staticmethod
    def ploVarianceOfCols(
        array2D: np.ndarray) -> None:
        """ Take a 2D array and plot the VARIANCE of each COLUMN """
        data = np.var(array2D,axis=1)
        plotSimpleLine(data,None,"Col Index","Variance Value")
        return None

    @staticmethod
    def plotMedianOfRows(
        array2D: np.ndarray) -> None:
        """ Take a 2D array and plot the MEDIAN of each ROW """
        data = np.median(array2D,axis=0)
        plotSimpleLine(data,None,"Row Index","Median Value")
        return None

    @staticmethod
    def plotMedianOfCols(
        array2D: np.ndarray) -> None:
        """ Take a 2D array and plot the MEDIAN of each COLUMN """
        data = np.median(array2D,axis=1)
        plotSimpleLine(data,None,"Col Index","Median Value")
        return None

    @staticmethod
    def plotMinMaxOfRows(
        array2D: np.ndarray) -> None:
        """ Take a 2D array and plot the MIN & MAX of each ROW """
        data = np.empty(shape=(2,array2D.shape[1]))
        data[0] = np.min(array2D,axis=0)
        data[1] = np.max(array2D,axis=0)
        plotSimpleLine(data.transpose(),None,"Row Index","Min & Max Value")
        return None

    @staticmethod
    def plotMinMaxOfCols(
        array2D: np.ndarray) -> None:
        """ Take a 2D array and plot the MIN & MAX of each COLUMN """
        data = np.empty(shape=(2,array2D.shape[0]))
        data[0] = np.min(array2D,axis=1)
        data[1] = np.max(array2D,axis=1)
        plotSimpleLine(data.transpose(),None,"Col Index","Min & Max Value")
        return None
    