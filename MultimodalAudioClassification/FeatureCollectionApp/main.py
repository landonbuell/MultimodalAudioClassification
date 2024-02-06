"""
Repo:       MultiModalAudioClassification
Solution:   MultiModalAudioClassification
Project:    FeautureCollection
File:       main.py

Author:     Landon Buell
Date:       June 2022
"""

    #### IMPORTS ####


import sys

import featureCollectionApp

    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Generate App Settings + App Instance
    settings = Administrative.AppSettings.developmentSettingsInstance()
    app = featureCollectionApp.FeatureCollectionApplication(settings)

    # Run Applicatin Execution Sequnce
    app.run()

    # Destroy the app + Exit
    sys.exit(0)
