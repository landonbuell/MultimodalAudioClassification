"""
Repo:       MultiModalAudioClassification
Solution:   MultiModalAudioClassification
Project:    FeautureCollection
File:       main.py

Author:     Landon Buell
Date:       June 2022
"""

    #### IMPORTS ####

import os
import sys

import scipy as sp

import appSettings
import featureCollectionApp

import sampleGeneratorPresets

    #### MAIN EXECUTABLE ####

def sessionSettings() -> appSettings.AppSettings:
    """ Return a settings instance """
    inputFilesHome = "C:\\Users\\lando\\Documents\\GitHub\\MultimodalAudioClassification\\InputFiles"
    inputFiles = [  #os.path.join(inputFilesHome,"Y1.csv"),
                    #os.path.join(inputFilesHome,"Y2.csv"),
                    #os.path.join(inputFilesHome,"Y3.csv"),
                    #os.path.join(inputFilesHome,"Y4.csv"), 
                    ]
    dataGenerators = [
            sampleGeneratorPresets.getUniformSquare(4096,0),
            sampleGeneratorPresets.getUniformCosine(4096,1),
        ]
    outputPath = "C:\\Users\\lando\\Documents\\audioFeatures\\simpleSignalsV4"
    settings = appSettings.AppSettings(inputFiles,dataGenerators,outputPath)
    return settings

if __name__ == "__main__":

    # Generate App Settings + App Instance
    settings = sessionSettings()
    app = featureCollectionApp.FeatureCollectionApplication(settings)

    # Run the application
    app.run()

    # Destroy the app + Exit
    sys.exit(0)
