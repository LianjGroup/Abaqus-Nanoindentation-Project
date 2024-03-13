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
import stage0_configs 
from math import *
import json
from datetime import datetime
import os
import prettytable
from optimizers.optimize import *

def main_iterative_calibration(info):

    # ---------------------------------------------------#
    #  Stage 4: Run iterative parameter calibration loop #
    # ---------------------------------------------------#

    logPath = info['logPath']
    resultPath = info['resultPath']

    material = info['material']
    optimizerName = info['optimizerName']
    paramConfig = info['paramConfig']
    deviationPercent = info['deviationPercent']

    targetCurves = info['targetCurves']
    targetCenters = info['targetCenters']
    objectives = info['objectives']
    paramConfig = info['paramConfig']
    optimizingInstance = info['optimizingInstance']

    combined_objective_value_to_param_FD_Curves = info["FD_Curves_dict"]['combined_objective_value_to_param_FD_Curves']
    iteration_objective_value_to_param_FD_Curves = info["FD_Curves_dict"]['iteration_objective_value_to_param_FD_Curves']
    

    # print("Hello")
    # time.sleep(180)
    
    nonconverging_combined_objective_value_to_param_FD_Curves = copy.deepcopy(combined_objective_value_to_param_FD_Curves)
    converging_combined_objective_value_to_param_FD_Curves = copy.deepcopy(combined_objective_value_to_param_FD_Curves)

    for objective in objectives:
        nonconverging_combined_objective_value_to_param_FD_Curves[objective] = filter_simulations(combined_objective_value_to_param_FD_Curves[objective], nonconverging=True)
        converging_combined_objective_value_to_param_FD_Curves[objective] = filter_simulations(combined_objective_value_to_param_FD_Curves[objective] , nonconverging=False)

    converged_sim_centers = copy.deepcopy(combined_objective_value_to_param_FD_Curves)
    for objective in objectives:
        for params, dispForce in combined_objective_value_to_param_FD_Curves[objective].items():
            simCenter = find_sim_center(dispForce)
            converged_sim_centers[objective][params] = simCenter
    
    last_simCenters = {}
    for objective in objectives:
        last_simCenters[objective] = list(converged_sim_centers[objective].values())[-1]

    printLog("=====================================================", logPath)
    for objective in objectives:
        printLog(f"The last converged sim centers for {objective} are: ", logPath)
        printLog(str(last_simCenters[objective]), logPath)

    # print("Hello")
    # time.sleep(180)
    
    printLog("=====================================================", logPath)
    printLog(f"Training the classifiers for the objectives", logPath)

    classifiers = train_classifiers(nonconverging_combined_objective_value_to_param_FD_Curves, 
                                    converging_combined_objective_value_to_param_FD_Curves, 
                                    paramConfig,
                                    objectives)
    
    printLog("=====================================================", logPath)
    printLog(f"Training the regressors for the objectives", logPath)
    regressionModels = train_linear_models(targetCenters, 
                        converging_combined_objective_value_to_param_FD_Curves, 
                        paramConfig, 
                        objectives)
        
    stopAllObjectives, stopAllObjectivesCheckObjectives = stopFD_MOO(targetCenters, 
                                                                     last_simCenters, 
                                                                     deviationPercent, 
                                                                     objectives)
    
    sim = SIMULATION(info)

    while not stopAllObjectives:
        print("=====================================================")
        print("The satisfying objectives are")
        for objective in objectives:
            X_deviation = abs(targetCenters[objective]['X'] - last_simCenters[objective]['X']) / targetCenters[objective]['X'] * 100
            Y_deviation = abs(targetCenters[objective]['Y'] - last_simCenters[objective]['Y']) / targetCenters[objective]['Y'] * 100
            print(f"{objective}: {stopAllObjectivesCheckObjectives[objective]}")
            print(f"X deviation: {X_deviation:.4f}%, Y deviation: {Y_deviation:.4f}%")
        # print("Hello")  
        # time.sleep(180)  
        exampleObjective = objectives[0]
        iterationIndex = len(iteration_objective_value_to_param_FD_Curves[exampleObjective]) + 1

        printLog("\n" + 60 * "#" + "\n", logPath)
        printLog(f"Running iteration {iterationIndex} for {optimizingInstance}" , logPath)
        printLog(f"The next predicted candidate parameters for simulation", logPath)
        printLog(f"======================================================", logPath)
        
        next_paramsDict = minimize_custom_loss_function(classifiers, 
                                                  regressionModels, 
                                                  paramConfig,
                                                  objectives)

        # next_paramsTuple = tuple(next_paramsDict.items())
       
        prettyPrint(next_paramsDict, paramConfig, logPath)

        time.sleep(60)

        printLog("Start running iteration simulation", logPath)
        
        objective_value_to_param_new_FD_Curves = sim.run_iteration_simulations(next_paramsDict, 
                                                                               iterationIndex)
        
        # Updating the combined and iteration FD curves smooth
        for objective in objectives:
            combined_objective_value_to_param_FD_Curves[objective].update(objective_value_to_param_new_FD_Curves[objective])
            iteration_objective_value_to_param_FD_Curves[objective].update(objective_value_to_param_new_FD_Curves[objective])

        loss_newIteration = {}

        for objective in objectives:            
            sim_dispForce = list(iteration_objective_value_to_param_FD_Curves[objective].values())[-1]
            simCenter = find_sim_center(sim_dispForce)
            targetCenter = targetCenters[objective]
            loss_newIteration[objective] = lossFD_SOO(targetCenter, simCenter)
        
        printLog(f"The loss of the new iteration is: ", logPath)
        printLog(str(loss_newIteration), logPath)

        # Saving the iteration data
        for objective in objectives:
            np.save(f"{resultPath}/{objective}/iteration/common/FD_Curves.npy", iteration_objective_value_to_param_FD_Curves[objective])

        #######################################################
        # Repeating all the steps above before the while loop #
        #######################################################
            
        nonconverging_combined_objective_value_to_param_FD_Curves = copy.deepcopy(combined_objective_value_to_param_FD_Curves)
        converging_combined_objective_value_to_param_FD_Curves = copy.deepcopy(combined_objective_value_to_param_FD_Curves)

        for objective in objectives:
            nonconverging_combined_objective_value_to_param_FD_Curves[objective] = filter_simulations(combined_objective_value_to_param_FD_Curves[objective], nonconverging=True)
            converging_combined_objective_value_to_param_FD_Curves[objective] = filter_simulations(combined_objective_value_to_param_FD_Curves[objective] , nonconverging=False)

        converged_sim_centers = copy.deepcopy(combined_objective_value_to_param_FD_Curves)
        for objective in objectives:
            for params, dispForce in combined_objective_value_to_param_FD_Curves[objective].items():
                simCenter = find_sim_center(dispForce)
                converged_sim_centers[objective][params] = simCenter
        
        last_simCenters = {}
        for objective in objectives:
            last_simCenters[objective] = list(converged_sim_centers[objective].values())[-1]

        printLog("=====================================================", logPath)
        for objective in objectives:
            printLog(f"The last converged sim centers for {objective} are: ", logPath)
            printLog(str(last_simCenters[objective]), logPath)

        printLog("=====================================================", logPath)
        printLog(f"Training the classifiers for the objectives", logPath)

        classifiers = train_classifiers(nonconverging_combined_objective_value_to_param_FD_Curves, 
                                        converging_combined_objective_value_to_param_FD_Curves, 
                                        paramConfig,
                                        objectives)
        
        printLog("=====================================================", logPath)
        printLog(f"Training the regressors for the objectives", logPath)
        regressionModels = train_linear_models(targetCenters, 
                            converging_combined_objective_value_to_param_FD_Curves, 
                            paramConfig, 
                            objectives)
            
        stopAllObjectives, stopAllObjectivesCheckObjectives = stopFD_MOO(targetCenters, 
                                                                        last_simCenters, 
                                                                        deviationPercent, 
                                                                        objectives)