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

    for objective in objectives:
        combined_objective_value_to_param_FD_Curves[objective].update(iteration_objective_value_to_param_FD_Curves[objective])
    
    FD_Curves_dict = {}
    
    FD_Curves_dict['initial_objective_value_to_param_FD_Curves'] = initial_objective_value_to_param_FD_Curves
    FD_Curves_dict['iteration_objective_value_to_param_FD_Curves'] = iteration_objective_value_to_param_FD_Curves
    FD_Curves_dict['combined_objective_value_to_param_FD_Curves'] = combined_objective_value_to_param_FD_Curves

    FD_Curves_dict['combined_param_to_objective_value_FD_Curves'] = reverseAsParamsToObjectives(combined_objective_value_to_param_FD_Curves, objectives)
    FD_Curves_dict['iteration_param_to_objective_value_FD_Curves'] = reverseAsParamsToObjectives(iteration_objective_value_to_param_FD_Curves, objectives)

    # print("Hello")
    # time.sleep(180)
    return FD_Curves_dict

    