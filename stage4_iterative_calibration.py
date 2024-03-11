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
    objectives = info['objectives']
    combined_objective_value_to_param_FD_Curves = info["FD_Curves_dict"]['combined_objective_value_to_param_FD_Curves']
    iteration_objective_value_to_param_FD_Curves = info["FD_Curves_dict"]['iteration_objective_value_to_param_FD_Curves']

    sim = SIMULATION(info)

    #np.save("combined_interpolated_param_to_geom_FD_Curves_smooth.npy", combined_interpolated_param_to_geom_FD_Curves_smooth)
    #print("Hello")
    #time.sleep(180)

    # np.save("targetCurves.npy", targetCurves)
    # print("Hello")
    # time.sleep(180)
    
    classifiers = train_nonconverging_classifiers(combined_objective_value_to_param_FD_Curves, objectives)
    time.sleep(180)
    
    while not stopFD_MOO(targetCurves, list(combined_interpolated_param_to_geom_FD_Curves_smooth.values())[-1], geometries, yieldingIndices, deviationPercent):

        iterationIndex = len(iteration_original_param_to_geom_FD_Curves_smooth) + 1
        exampleGeometry = geometries[0]

        # Weighted single objective optimization strategy:
        if optimizerName == "BO":
            geometryWeights = MOO_calculate_geometries_weight(targetCurves, geometries)
            printLog("The weights for the geometries are: ", logPath)
            printLog(str(geometryWeights), logPath)
            

            MOO_write_BO_json_log(combined_interpolated_param_to_geom_FD_Curves_smooth, targetCurves, geometries, geometryWeights, yieldingIndices, paramConfig,iterationIndex)
            BO_instance = BO(info)
            BO_instance.initializeOptimizer(lossFunction=None, param_bounds=param_bounds, loadingProgress=True)
            next_paramDict = BO_instance.suggest()
            next_paramDict = rescale_paramsDict(next_paramDict, paramConfig)
            
        if optimizerName == "BOTORCH":
            pareto_front = MOO_suggest_BOTORCH(combined_interpolated_param_to_geom_FD_Curves_smooth, targetCurves, geometries, yieldingIndices, paramConfig,iterationIndex)
            # Select a random point from the pareto front
            next_paramDict = pareto_front[0]

        #print(len(iteration_interpolated_FD_Curves_smooth))
        printLog("\n" + 60 * "#" + "\n", logPath)
        printLog(f"Running iteration {iterationIndex} for {material}_{hardeningLaw}_curve{curveIndex}" , logPath)
        printLog(f"The next candidate {hardeningLaw} parameters predicted by {optimizerName}", logPath)
        prettyPrint(next_paramDict, paramConfig, logPath)

        time.sleep(30)
        printLog("Start running iteration simulation", logPath)
        
        geom_to_param_new_FD_Curves, geom_to_param_new_flowCurves = sim.run_iteration_simulations(next_paramDict, iterationIndex)
        
        geom_to_param_new_FD_Curves_unsmooth = copy.deepcopy(geom_to_param_new_FD_Curves)
        geom_to_param_new_FD_Curves_smooth = copy.deepcopy(geom_to_param_new_FD_Curves)
        new_param = list(geom_to_param_new_FD_Curves[exampleGeometry].keys())[0]
        
        for geometry in geometries:
            geom_to_param_new_FD_Curves_smooth[geometry][new_param]['force'] = smoothing_force(geom_to_param_new_FD_Curves_unsmooth[geometry][new_param]['force'], startIndex=20, endIndex=90, iter=20000)
        
        # Updating the combined FD curves smooth
        for geometry in geometries:
            combined_original_geom_to_param_FD_Curves_smooth[geometry].update(geom_to_param_new_FD_Curves_smooth[geometry])
            combined_interpolated_geom_to_param_FD_Curves_smooth[geometry] = interpolating_FD_Curves(combined_original_geom_to_param_FD_Curves_smooth[geometry], targetCurves[geometry])
        
        # Updating the iteration FD curves smooth
        for geometry in geometries:
            iteration_original_geom_to_param_FD_Curves_smooth[geometry].update(geom_to_param_new_FD_Curves_smooth[geometry])
            iteration_interpolated_geom_to_param_FD_Curves_smooth[geometry] = interpolating_FD_Curves(iteration_original_geom_to_param_FD_Curves_smooth[geometry], targetCurves[geometry])
        
        # Updating the iteration FD curves unsmooth
        for geometry in geometries:
            iteration_original_geom_to_param_FD_Curves_unsmooth[geometry].update(geom_to_param_new_FD_Curves_unsmooth[geometry])
        
        # Updating the original flow curves
        for geometry in geometries:
            combined_original_geom_to_param_flowCurves[geometry].update(geom_to_param_new_flowCurves[geometry])
            iteration_original_geom_to_param_flowCurves[geometry].update(geom_to_param_new_flowCurves[geometry])
        
        # Updating the param_to_geom data
        combined_interpolated_param_to_geom_FD_Curves_smooth = reverseAsParamsToGeometries(combined_interpolated_geom_to_param_FD_Curves_smooth, geometries)
        iteration_original_param_to_geom_FD_Curves_smooth = reverseAsParamsToGeometries(iteration_original_geom_to_param_FD_Curves_smooth, geometries)

        loss_newIteration = {}
        for geometry in geometries:
            yieldingIndex = yieldingIndices[geometry]
            
            simForce = list(iteration_interpolated_geom_to_param_FD_Curves_smooth[geometry].values())[-1]['force'][yieldingIndex:]
            simDisplacement = list(iteration_interpolated_geom_to_param_FD_Curves_smooth[geometry].values())[-1]['displacement'][yieldingIndex:]
            targetForce = targetCurves[geometry]['force'][yieldingIndex:]
            targetDisplacement = targetCurves[geometry]['displacement'][yieldingIndex:]
            interpolated_simForce = interpolatingForce(simDisplacement, simForce, targetDisplacement)
            loss_newIteration[geometry] = round(lossFD(targetDisplacement, targetForce, interpolated_simForce,iterationIndex), 3)
        
        printLog(f"The loss of the new iteration is: ", logPath)
        printLog(str(loss_newIteration), logPath)

        # Saving the iteration data
        for geometry in geometries:
            np.save(f"{resultPath}/{geometry}/iteration/common/FD_Curves_unsmooth.npy", iteration_original_geom_to_param_FD_Curves_unsmooth[geometry])
            np.save(f"{resultPath}/{geometry}/iteration/common/FD_Curves_smooth.npy", iteration_original_geom_to_param_FD_Curves_smooth[geometry])
            np.save(f"{resultPath}/{geometry}/iteration/common/flowCurves.npy", iteration_original_geom_to_param_flowCurves[geometry])