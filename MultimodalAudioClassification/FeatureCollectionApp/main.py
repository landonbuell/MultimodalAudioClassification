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

import appSettings
import featureCollectionApp

    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Generate App Settings + App Instance
    settings = appSettings.AppSettings.developmentSettingsInstance()
    app = featureCollectionApp.FeatureCollectionApplication(settings)

    # Run the application
    app.run()

    # Destroy the app + Exit
    sys.exit(0)
