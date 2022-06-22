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

import Administrative

    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Generate App Settings + App Instance
    settings = Administrative.AppSettings.developmentSettingsInstance()
    app = Administrative.FeatureCollectionApp.constructApp(settings)

    # Run Applicatin Execution Sequnce
    app.startup()
    app.execute()
    app.shutdown()

    # Destroy the app + Exit
    Administrative.FeatureCollectionApp.destroyApp()
    sys.exit(0)
