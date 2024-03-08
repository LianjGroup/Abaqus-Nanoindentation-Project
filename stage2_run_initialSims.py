import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from modules.SIM import *
from modules.IO import *
from modules.calculation import *
from optimizers.optimize import *
from modules.stoploss import *
from stage0_configs import * 
import stage0_configs 
import stage1_prepare_targetCurve
from stage1_prepare_targetCurve import *
from math import *
import json
from datetime import datetime
import os
import prettytable

def main_run_initialSims(info):

    # ---------------------------------------#
    #   Step 2: Running initial simulations  #
    # ---------------------------------------#
    
    logPath = info['logPath']
    resultPath = info['resultPath']
    simPath = info['simPath']
    templatePath = info['templatePath'] 

    objectives = info['objectives']
    numberOfInitialSims = info['numberOfInitialSims']
    maxConcurrentSimNumber = info['maxConcurrentSimNumber']
    deleteSimOutputs = info['deleteSimOutputs']
    
    if os.path.exists(f"{resultPath}/initialParameters.npy"):
        initialParams = np.load(f"{resultPath}/initialParameters.npy", allow_pickle=True).tolist()
        info['numberOfInitialSims'] = len(initialParams)
        print(f"initialParameters.npy exists in {resultPath}\n")
        print(f"Number of initial simulations: {len(initialParams)}. Parameters is loaded from {resultPath}\n")
        print("If you want to regenerate the initialParameters.npy, please delete initialParameters.npy and run the program again\n")
    else:
        
        sim = SIMULATION(info)
        initialParams = sim.sobol_sequence_sampling()
        # parameters = sim.latin_hypercube_sampling()
        print(f"No initial parameters.npy exists in {resultPath}\n")
        print("Program will generate the parameters.npy\n")
        print(f"Number of initial simulations: {len(initialParams)}. Parameters is saved in {resultPath}\n")
        # print(parameters)
        # time.sleep(180)
        np.save(f"{resultPath}/initialParameters.npy", initialParams)

    # print("Checkpoint")
    # time.sleep(180)

    for objective in objectives:
        infoCopy = copy.deepcopy(info)
        resultPathObjective = f"{resultPath}/{objective}"
        simPathObjective = f"{simPath}/{objective}"
        templatePathObjective = f"{templatePath}/{objective}"
        infoCopy['resultPath'] = resultPathObjective
        infoCopy['simPath'] = simPathObjective
        infoCopy['templatePath'] = templatePathObjective
    
        sim = SIMULATION(infoCopy) 

        if os.path.exists(f"{resultPathObjective}/initial/common/FD_Curves.npy"):
            printLog("=======================================================================", logPath)
            printLog(f"FD_Curves.npy for {objective} already exist", logPath)
            numberOfInitialSims = len(np.load(f"{resultPathObjective}/initial/common/FD_Curves.npy", allow_pickle=True).tolist())
            printLog(f"Number of initial simulations for {objective}: {numberOfInitialSims} FD curves", logPath)
        else:
            if maxConcurrentSimNumber == "max" or maxConcurrentSimNumber >= len(initialParams):
                currentIndices = range(1, len(initialParams) + 1)
                batchNumber = "all"
                printLog("=======================================================================", logPath)
                printLog(f"There are no FD_Curves.npy for {objective}", logPath)
                printLog(f"Program starts running the initial simulations for {objective}", logPath)   
                sim.run_initial_simulations(initialParams=initialParams, currentIndices=currentIndices, batchNumber=batchNumber)
                # time.sleep(180)
                printLog(f"Initial simulations for {objective} have completed", logPath)
            else:
                for i in range(0, len(initialParams), maxConcurrentSimNumber):
                    batchNumber = int(i/maxConcurrentSimNumber + 1)
                    currentIndices = range(i + 1, i + maxConcurrentSimNumber + 1)

                    if not os.path.exists(f"{resultPathObjective}/initial/common/FD_Curves_batch_{batchNumber}.npy"):
                        printLog("=======================================================================", logPath)
                        printLog(f"There are no FD_Curves_batch_{batchNumber}.npy for {objective}", logPath)
                        printLog(f"Program starts running the initial simulations of batch number {batchNumber} for {objective}", logPath)   
                        if i + maxConcurrentSimNumber > len(initialParams):
                            sub_initialParams = initialParams[i:]
                        else:
                            sub_initialParams = initialParams[i:i+maxConcurrentSimNumber]
                        sim.run_initial_simulations(initialParams=sub_initialParams, currentIndices=currentIndices, batchNumber=batchNumber)
                        printLog(f"Initial simulations of batch {batchNumber} for {objective} have completed", logPath)
                    else: 
                        printLog("=======================================================================", logPath)
                        printLog(f"FD_Curves_batch_{batchNumber}.npy for {objective} already exists", logPath)
                    
                # Now we combine the batches of simulations together into FD_Curves.npy
                printLog("=======================================================================", logPath)
                printLog(f"Start concatenating the batches of initial guesses together", logPath)

                FD_Curves = {}
                for i in range(0, len(initialParams), maxConcurrentSimNumber):
                    batchNumber = int(i/maxConcurrentSimNumber + 1)
                    FD_Curves_batch = np.load(f"{resultPathObjective}/initial/common/FD_Curves_batch_{batchNumber}.npy", allow_pickle=True).tolist()
                    FD_Curves.update(FD_Curves_batch)
                np.save(f"{resultPathObjective}/initial/common/FD_Curves.npy", FD_Curves)
                numberOfInitialSims = len(FD_Curves)
                printLog("Finish concatenating the batches of initial guesses together", logPath)
                printLog(f"Number of initial simulations for {objective}: {numberOfInitialSims} FD curves", logPath)

if __name__ == "__main__":

    info = stage0_configs.main_config()
    
    logPath = info['logPath']
    deviationPercent = info['deviationPercent']

    targetCurves, targetCenters = stage1_prepare_targetCurve.main_prepare_targetCurve(info)
    info['targetCurves'] = targetCurves
    info['targetCenters'] = targetCenters

    main_run_initialSims(info)