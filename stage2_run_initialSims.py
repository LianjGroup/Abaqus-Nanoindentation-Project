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
from optimizers.BO import *
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
    
    projectPath = info['projectPath']
    logPath = info['logPath']
    resultPath = info['resultPath']
    simPath = info['simPath']
    targetPath = info['targetPath']
    templatePath = info['templatePath'] 
    material = info['material']
    optimizerName = info['optimizerName']
    hardeningLaw = info['hardeningLaw']
    paramConfig = info['paramConfig']
    geometries = info['geometries']
    deviationPercent = info['deviationPercent']
    numberOfInitialSims = info['numberOfInitialSims']
    maxTargetDisplacements = info['maxTargetDisplacements']

    if os.path.exists(f"{resultPath}/parameters.npy"):
        parameters = np.load(f"{resultPath}/parameters.npy", allow_pickle=True).tolist()
        info['numberOfInitialSims'] = len(parameters)
        print(f"Initial parameters.npy exists in {resultPath}\n")
        print("If you want to regenerate the parameters.npy, please delete parameters.npy and run the program again\n")
    else:
        parameters = sim.latin_hypercube_sampling(geometry)
        print(f"No initial parameters.npy exists in {resultPath}\n")
        print("Program will generate the parameters.npy\n")

    #print("Checkpoint")
    #time.sleep(180)

    for geometry in geometries:
        infoCopy = copy.deepcopy(info)
        resultPathGeometry = f"{resultPath}/{geometry}"
        simPathGeometry = f"{simPath}/{geometry}"
        templatePathGeometry = f"{templatePath}/{geometry}"
        infoCopy['resultPath'] = resultPathGeometry
        infoCopy['simPath'] = simPathGeometry
        infoCopy['templatePath'] = templatePathGeometry
        infoCopy['maxTargetDisplacement'] = maxTargetDisplacements[geometry]
        sim = SIMULATION(infoCopy) 

        if not os.path.exists(f"{resultPathGeometry}/initial/common/FD_Curves_unsmooth.npy"):
            printLog("=======================================================================", logPath)
            printLog(f"There are no initial simulations for {geometry} geometry", logPath)
            printLog(f"Program starts running the initial simulations for {geometry} geometry", logPath)
            sim.run_initial_simulations(parameters)
            printLog(f"Initial simulations for {geometry} geometry have completed", logPath)
        else: 
            printLog("=======================================================================", logPath)
            printLog(f"Initial simulations for {geometry} geometry already exist", logPath)
            numberOfInitialSims = len(np.load(f"{resultPathGeometry}/initial/common/FD_Curves_unsmooth.npy", allow_pickle=True).tolist())
            printLog(f"Number of initial simulations for {geometry} geometry: {numberOfInitialSims} FD curves", logPath)

if __name__ == "__main__":
    info = stage0_configs.main_config()
    targetCurves, maxTargetDisplacements = stage1_prepare_targetCurve.main_prepare_targetCurve(info)
    info['targetCurves'] = targetCurves
    info['maxTargetDisplacements'] = maxTargetDisplacements
    main_run_initialSims(info)