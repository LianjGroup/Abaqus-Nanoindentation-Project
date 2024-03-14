import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from modules.IO import *
from modules.calculation import *
from optimizers.optimize import *
from modules.stoploss import *
import stage0_configs
from math import *
import json
from datetime import datetime
import os
import prettytable
import copy

def main_prepare_simCurves(info):

    # ---------------------------------------------------------------------#
    #   Step 3: Preparing FD curves from initial and iteration simulations #
    # ---------------------------------------------------------------------#
    

    logPath = info['logPath']
    resultPath = info['resultPath']
    simPath = info['simPath']
    templatePath = info['templatePath'] 

    objectives = info['objectives']
    numberOfInitialSims = info['numberOfInitialSims']
    maxConcurrentSimNumber = info['maxConcurrentSimNumber']
    deleteSimOutputs = info['deleteSimOutputs']

    # Loading initial simulations
    exampleObjective = objectives[0]

    initial_objective_value_to_param_FD_Curves = {}

    for objective in objectives:
        initial_objective_value_to_param_FD_Curves[objective] = np.load(f"{resultPath}/{objective}/initial/common/FD_Curves.npy", allow_pickle=True).tolist()
    
    iteration_objective_value_to_param_FD_Curves = {}
    
    # Check if there are any iteration simulations
    if not os.path.exists(f"{resultPath}/{exampleObjective}/iteration/common/FD_Curves.npy"):
        printLog("There are no iteration simulations. Program starts running the iteration simulations", logPath)
        for objective in objectives:
            iteration_objective_value_to_param_FD_Curves[objective] = {}
    else:
        printLog("Iteration simulations exist", logPath)
        numberOfIterationSims = len(np.load(f"{resultPath}/{exampleObjective}/iteration/common/FD_Curves.npy", allow_pickle=True).tolist())
        printLog(f"Number of iteration simulations: {numberOfIterationSims} FD curves", logPath)
        for objective in objectives:
            iteration_objective_value_to_param_FD_Curves[objective] = np.load(f"{resultPath}/{objective}/iteration/common/FD_Curves.npy", allow_pickle=True).tolist()

    combined_objective_value_to_param_FD_Curves = copy.deepcopy(initial_objective_value_to_param_FD_Curves)
    
    iteration_objective_value_to_param_FD_Curves_copy = copy.deepcopy(iteration_objective_value_to_param_FD_Curves)
    
    for objective in objectives:
        combined_objective_value_to_param_FD_Curves[objective].update(iteration_objective_value_to_param_FD_Curves_copy[objective])
    
    # Converting units for the FD curves
    # Displacement is in nanometers and force is in micro Newtons in experiment
    # while simulation returns measurements in meters and Newtons (very small number)
    for objective in objectives:
        for paramsTuple, dispForce in initial_objective_value_to_param_FD_Curves[objective].items():
            initial_objective_value_to_param_FD_Curves[objective][paramsTuple]["displacement"] *= 1e9
            initial_objective_value_to_param_FD_Curves[objective][paramsTuple]["force"] *= 1e6
        
        for paramsTuple, dispForce in iteration_objective_value_to_param_FD_Curves[objective].items():
            iteration_objective_value_to_param_FD_Curves[objective][paramsTuple]["displacement"] *= 1e9
            iteration_objective_value_to_param_FD_Curves[objective][paramsTuple]["force"] *= 1e6
        
        for paramsTuple, dispForce in combined_objective_value_to_param_FD_Curves[objective].items():
            combined_objective_value_to_param_FD_Curves[objective][paramsTuple]["displacement"] *= 1e9
            combined_objective_value_to_param_FD_Curves[objective][paramsTuple]["force"] *= 1e6
    
    FD_Curves_dict = {}
    
    FD_Curves_dict['initial_objective_value_to_param_FD_Curves'] = initial_objective_value_to_param_FD_Curves
    FD_Curves_dict['iteration_objective_value_to_param_FD_Curves'] = iteration_objective_value_to_param_FD_Curves
    FD_Curves_dict['combined_objective_value_to_param_FD_Curves'] = combined_objective_value_to_param_FD_Curves

    # print("Hello")
    # time.sleep(180)
    return FD_Curves_dict

    