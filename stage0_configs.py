import os
import time
import pandas as pd
import numpy as np
from time import sleep
from prettytable import PrettyTable
from stage0_initialize_directory import *
from modules.IO import *
from modules.calculation import *
from optimizers.optimize import *

import copy

############################################################
#                                                          #
#    ABAQUS NANOINDENTATION CPFEM PARAMETERS CALIBRATION   #
#   Tools required: Abaqus and Finnish Supercomputer CSC   #
#                                                          #
############################################################

# ------------------------------------#
#   Stage 0: Recording configurations #
# ------------------------------------#

def main_config():

    #########################
    # Global configurations #
    #########################

    globalConfig = pd.read_excel("configs/global_config.xlsx", nrows=1, engine="openpyxl")
    globalConfig = globalConfig.T.to_dict()[0]
    SLURM_iteration = globalConfig["SLURM_iteration"]
    numberOfInitialSims = globalConfig["numberOfInitialSims"]
    initialSimsSpacing = globalConfig["initialSimsSpacing"]
    maxConcurrentSimNumber = globalConfig["maxConcurrentSimNumber"]
    if maxConcurrentSimNumber != "max":
        maxConcurrentSimNumber = int(maxConcurrentSimNumber)
    material = globalConfig["material"]
    CPLaw = globalConfig["CPLaw"]
    grains = globalConfig["grains"]
    strainRates = globalConfig["strainRates"]
    optimizerName = globalConfig["optimizerName"]
    deviationPercentX = globalConfig["deviationPercentX"]
    deviationPercentY = globalConfig["deviationPercentY"]
    deviationPercent = {"X": deviationPercentX, "Y": deviationPercentY}

    deleteSimOutputs = globalConfig["deleteSimOutputs"]
    
    grains = grains.split(";")
    strainRates = strainRates.split(";")

    (
        optimizingInstance,
        objectives,
        projectPath, 
        logPath, 
        paramInfoPath, 
        resultPath, 
        simPath, 
        templatePath, 
        targetPath
    ) = initialize_directory(material, CPLaw, grains, strainRates)

    ##################################
    # Parameter bound configurations #
    ##################################

    paramConfig = pd.read_excel(f"{paramInfoPath}/paramInfo.xlsx", engine="openpyxl")
    paramConfig.set_index("parameter", inplace=True)
    paramConfig = paramConfig.T.to_dict()
    for param in paramConfig:
        paramConfig[param]['exponent'] = float(paramConfig[param]['exponent'])

    ###########################
    # Information declaration #
    ###########################

    info = {
        'projectPath': projectPath,
        'logPath': logPath,
        'paramInfoPath': paramInfoPath,
        'resultPath': resultPath,
        'simPath': simPath,
        'targetPath': targetPath,
        'templatePath': templatePath,
        'SLURM_iteration': SLURM_iteration,
        'numberOfInitialSims': numberOfInitialSims,
        'initialSimsSpacing': initialSimsSpacing,
        'maxConcurrentSimNumber': maxConcurrentSimNumber,
        'deleteSimOutputs': deleteSimOutputs,
        'material': material,
        'CPLaw': CPLaw,
        'grains': grains,
        'objectives': objectives,
        'optimizingInstance': optimizingInstance,
        'strainRates': strainRates,
        'optimizerName': optimizerName,
        'paramConfig': paramConfig,
        'deviationPercent': deviationPercent,
    }

    ###############################################
    #  Printing the configurations to the console #
    ###############################################

    printLog(f"\nWelcome to Abaqus nanoindentation CP param calibration project\n\n", logPath)
    printLog(f"The configurations you have chosen: \n", logPath)
    
    logTable = PrettyTable()

    logTable.field_names = ["Global Configs", "User choice"]
    logTable.add_row(["SLURM iteration", SLURM_iteration])
    logTable.add_row(["Number of initial sims", numberOfInitialSims])
    logTable.add_row(["Initial sims spacing", initialSimsSpacing])
    logTable.add_row(["Max concurrent sim number", maxConcurrentSimNumber])
    logTable.add_row(["Delete output sims", deleteSimOutputs])
    logTable.add_row(["Material", material])
    CPLaw_names = {"PH": "Phenomenological law", "DB": "Dislocation-based law"}
    logTable.add_row(["CP law", CPLaw_names[CPLaw]])

    grainString = ";".join(grains)
    strainRateString = ";".join(strainRates)
    logTable.add_row(["Grains number", grainString])
    logTable.add_row(["Strain rates", strainRateString])

    logTable.add_row(["Optimizer name", optimizerName])
    logTable.add_row(["Deviation percent X", f"{deviationPercentX}"])
    logTable.add_row(["Deviation percent Y", f"{deviationPercentY}"])

    printLog(logTable.get_string() + "\n", logPath)

    printLog("Generating necessary directories\n", logPath)
    printLog(f"The path to your main project folder is\n", logPath)
    printLog(f"{projectPath}\n", logPath)

    #############################
    # Returning the information #
    #############################
    return info

if __name__ == "__main__":
    main_config()