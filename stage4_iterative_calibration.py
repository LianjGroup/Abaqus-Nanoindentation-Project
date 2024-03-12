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
    sim = SIMULATION(info)

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

        #printLog(f"The weights for the {objective} are: ", logPath)
        
    stopAllObjectives, stopAllObjectivesCheckObjectives = stopFD_MOO(targetCenters, 
                                                                     last_simCenters, 
                                                                     deviationPercent, 
                                                                     objectives)
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
        printLog(f"The next predictd candidate parameters for simulation", logPath)
        printLog(f"=====================================================", logPath)
        
        next_paramDict = minimize_custom_loss_function(classifiers, 
                                                  regressionModels, 
                                                  paramConfig,
                                                  objectives)

        prettyPrint(next_paramDict, paramConfig, logPath)

        #print(optimal_X)
        time.sleep(180)


        printLog("Start running iteration simulation", logPath)
        
        geom_to_param_new_FD_Curves, geom_to_param_new_flowCurves = sim.run_iteration_simulations(next_paramDict, iterationIndex)
        
        geom_to_param_new_FD_Curves_unsmooth = copy.deepcopy(geom_to_param_new_FD_Curves)
        geom_to_param_new_FD_Curves_smooth = copy.deepcopy(geom_to_param_new_FD_Curves)
        new_param = list(geom_to_param_new_FD_Curves[exampleobjective].keys())[0]
        
        for objective in objectives:
            geom_to_param_new_FD_Curves_smooth[objective][new_param]['force'] = smoothing_force(geom_to_param_new_FD_Curves_unsmooth[objective][new_param]['force'], startIndex=20, endIndex=90, iter=20000)
        
        # Updating the combined FD curves smooth
        for objective in objectives:
            combined_original_geom_to_param_FD_Curves_smooth[objective].update(geom_to_param_new_FD_Curves_smooth[objective])
            combined_interpolated_geom_to_param_FD_Curves_smooth[objective] = interpolating_FD_Curves(combined_original_geom_to_param_FD_Curves_smooth[objective], targetCurves[objective])
        
        # Updating the iteration FD curves smooth
        for objective in objectives:
            iteration_original_geom_to_param_FD_Curves_smooth[objective].update(geom_to_param_new_FD_Curves_smooth[objective])
            iteration_interpolated_geom_to_param_FD_Curves_smooth[objective] = interpolating_FD_Curves(iteration_original_geom_to_param_FD_Curves_smooth[objective], targetCurves[objective])
        
        # Updating the iteration FD curves unsmooth
        for objective in objectives:
            iteration_original_geom_to_param_FD_Curves_unsmooth[objective].update(geom_to_param_new_FD_Curves_unsmooth[objective])
        
        # Updating the original flow curves
        for objective in objectives:
            combined_original_geom_to_param_flowCurves[objective].update(geom_to_param_new_flowCurves[objective])
            iteration_original_geom_to_param_flowCurves[objective].update(geom_to_param_new_flowCurves[objective])
        
        # Updating the param_to_geom data
        combined_interpolated_param_to_geom_FD_Curves_smooth = reverseAsParamsToobjectives(combined_interpolated_geom_to_param_FD_Curves_smooth, objectives)
        iteration_original_param_to_geom_FD_Curves_smooth = reverseAsParamsToobjectives(iteration_original_geom_to_param_FD_Curves_smooth, objectives)

        loss_newIteration = {}
        for objective in objectives:
            yieldingIndex = yieldingIndices[objective]
            
            simForce = list(iteration_interpolated_geom_to_param_FD_Curves_smooth[objective].values())[-1]['force'][yieldingIndex:]
            simDisplacement = list(iteration_interpolated_geom_to_param_FD_Curves_smooth[objective].values())[-1]['displacement'][yieldingIndex:]
            targetForce = targetCurves[objective]['force'][yieldingIndex:]
            targetDisplacement = targetCurves[objective]['displacement'][yieldingIndex:]
            interpolated_simForce = interpolatingForce(simDisplacement, simForce, targetDisplacement)
            loss_newIteration[objective] = round(lossFD(targetDisplacement, targetForce, interpolated_simForce,iterationIndex), 3)
        
        printLog(f"The loss of the new iteration is: ", logPath)
        printLog(str(loss_newIteration), logPath)

        # Saving the iteration data
        for objective in objectives:
            np.save(f"{resultPath}/{objective}/iteration/common/FD_Curves_unsmooth.npy", iteration_original_geom_to_param_FD_Curves_unsmooth[objective])
