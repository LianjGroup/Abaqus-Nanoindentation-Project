import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from modules.SIM import *
from modules.hardeningLaws import *
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

    grains = info['grains']
    strainRates = info['strainRates']
    numberOfInitialSims = info['numberOfInitialSims']
    
    
    if os.path.exists(f"{resultPath}/parameters.npy"):
        parameters = np.load(f"{resultPath}/parameters.npy", allow_pickle=True).tolist()
        info['numberOfInitialSims'] = len(parameters)
        print(f"Initial parameters.npy exists in {resultPath}\n")
        print(f"Number of initial simulations: {len(parameters)}. Parameters is loaded from {resultPath}\n")
        print("If you want to regenerate the parameters.npy, please delete parameters.npy and run the program again\n")
    else:
        
        sim = SIMULATION(info)
        parameters = sim.sobol_sequence_sampling()
        # parameters = sim.latin_hypercube_sampling()
        print(f"No initial parameters.npy exists in {resultPath}\n")
        print("Program will generate the parameters.npy\n")
        print(f"Number of initial simulations: {len(parameters)}. Parameters is saved in {resultPath}\n")
        # print(parameters)
        # time.sleep(180)
        np.save(f"{resultPath}/parameters.npy", parameters)

    # print("Checkpoint")
    # time.sleep(180)

    for grain in grains:
        for strainRate in strainRates:
            infoCopy = copy.deepcopy(info)
            resultPathGrainSR = f"{resultPath}/grain_{grain}_sr_{strainRate}"
            simPathGrainSR = f"{simPath}/grain_{grain}_sr_{strainRate}"
            templatePathGrainSR = f"{templatePath}/grain_{grain}_sr_{strainRate}"
            infoCopy['resultPath'] = resultPathGrainSR
            infoCopy['simPath'] = simPathGrainSR
            infoCopy['templatePath'] = templatePathGrainSR
     
            sim = SIMULATION(infoCopy) 

            if not os.path.exists(f"{resultPathGrainSR}/initial/common/FD_Curves.npy"):
                printLog("=======================================================================", logPath)
                printLog(f"There are no initial simulations for grain {grain}, strain rate {strainRate}", logPath)
                printLog(f"Program starts running the initial simulations for grain {grain}, strain rate {strainRate}", logPath)
                sim.run_initial_simulations(parameters)
                time.sleep(180)
                printLog(f"Initial simulations for grain {grain}, strain rate {strainRate} have completed", logPath)
            else: 
                printLog("=======================================================================", logPath)
                printLog(f"Initial simulations for grain {grain}, strain rate {strainRate} already exist", logPath)
                numberOfInitialSims = len(np.load(f"{resultPathGrainSR}/initial/common/FD_Curves_unsmooth.npy", allow_pickle=True).tolist())
                printLog(f"Number of initial simulations for grain {grain}, strain rate {strainRate}: {numberOfInitialSims} FD curves", logPath)

if __name__ == "__main__":

    info = stage0_configs.main_config()
    
    logPath = info['logPath']
    deviationPercent = info['deviationPercent']

    targetCurves, targetCenters = stage1_prepare_targetCurve.main_prepare_targetCurve(info)
    info['targetCurves'] = targetCurves
    info['targetCenters'] = targetCenters

    main_run_initialSims(info)