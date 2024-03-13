import numpy as np
import pandas as pd

import copy
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from modules.stoploss import *
import time


def parseBoundsBO(paramInfo):
    paramBounds = {}
    for param in paramInfo:
        paramBounds[param] = (paramInfo[param]['lowerBound'], paramInfo[param]['upperBound'])
    return paramBounds

# def is_directory_empty(directory_path):
#     return len(os.listdir(directory_path)) == 0

def smoothing_force(force, startIndex, endIndex, iter=20000):
    smooth_force = copy.deepcopy(force)
    for i in range(iter):
        smooth_force = savgol_filter(smooth_force[startIndex:endIndex], 
                                    window_length=5, 
                                    polyorder=3,
                                    mode='interp',
                                    #mode='nearest',
                                    #mode='mirror',
                                    #mode='wrap',
                                    #mode='constant',
                                    #deriv=0,
                                    delta=1
                                    )
        smooth_force = np.concatenate((force[0:startIndex], smooth_force, force[endIndex:]))
    return smooth_force

def interpolatingForce(simDisplacement, simForce, targetDisplacement):
    interpolatingFunction = interp1d(simDisplacement, simForce, fill_value='extrapolate')
    # Interpolate the force
    interpolatedSimForce = interpolatingFunction(targetDisplacement)
    return interpolatedSimForce

def interpolating_FD_Curves(FD_Curves, targetCurve):
    # Interpolate the force from FD_Curves to the target curve
    # FD_Curves is a dictionaries
    # where each element is of form (parameterTuples) => {"displacement": <np.array>, "force": <np.array>}
    # targetCurve is a dictionary of form {"displacement": <np.array>, "force": <np.array>}

    # Create interp1d fitting from scipy
    FD_Curves_copy = copy.deepcopy(FD_Curves)
    for paramsTuple, dispforce in FD_Curves_copy.items():
        simDisp = dispforce["displacement"]
        simForce = dispforce["force"]
        targetDisp = targetCurve["displacement"]
        # Interpolate the force
        FD_Curves_copy[paramsTuple]["force"] = interpolatingForce(simDisp, simForce, targetDisp)
        FD_Curves_copy[paramsTuple]["displacement"] = targetDisp
    return FD_Curves_copy

def interpolatingStress(simStrain, simStress, targetStrain):
    interpolatingFunction = interp1d(simStrain, simStress, fill_value='extrapolate')
    # Interpolate the stress
    interpolatedSimStress = interpolatingFunction(targetStrain)
    return interpolatedSimStress

def interpolating_flowCurves(flowCurves, targetCurve):
    flowCurves_copy = copy.deepcopy(flowCurves)
    for paramsTuple, strainstress in flowCurves_copy.items():
        simStrain = strainstress["strain"]
        simStress = strainstress["stress"]
        targetStrain = targetCurve["strain"]
        # Interpolate the force
        flowCurves_copy[paramsTuple]["stress"] = interpolatingStress(simStrain, simStress, targetStrain)
        flowCurves_copy[paramsTuple]["strain"] = targetStrain
    return flowCurves_copy

def rescale_paramsDict(paramsDict, paramConfig):
    rescaled_paramsDict = {}
    for param, value in paramsDict.items():
        rescaled_paramsDict[param] = value * paramConfig[param]['exponent']
    return rescaled_paramsDict

def reverseAsParamsToObjectives(curves, objectives):
    exampleObjective = objectives[0]
    reverseCurves = {}
    for paramsTuple in curves[exampleObjective]:
        reverseCurves[paramsTuple] = {}
        for objective in objectives:
            reverseCurves[paramsTuple][objective] = curves[objective][paramsTuple]
    return reverseCurves

# Returning list of 0 and 1, 1 if the simulation is nonconverging
def check_convergence(FD_Curves):
    
    maxDispCorrespondingToMaxForce = -np.inf

    for params, dispForce in FD_Curves.items():
        disp = dispForce["displacement"]
        force = dispForce["force"]
        if maxDispCorrespondingToMaxForce < disp[np.argmax(force)]: 
            maxDispCorrespondingToMaxForce = disp[np.argmax(force)]
    
    nonconverging_flags = []
    for i, (params, dispForce) in enumerate(FD_Curves.items()):
        disp = dispForce["displacement"]
        force = dispForce["force"]
        dispCorrespondingToMaxForce = disp[np.argmax(force)]
        if dispCorrespondingToMaxForce < maxDispCorrespondingToMaxForce - 2:
            nonconverging_flags.append(1)
        else:
            nonconverging_flags.append(0)
    return nonconverging_flags

def filter_simulations(FD_Curves, nonconverging_filter):

    FD_Curves_copy = {}
    nonconverging_flags = check_convergence(FD_Curves)
    #print(nonconverging_flags)
    if nonconverging_filter == True:
        for i, (params, dispForce) in enumerate(FD_Curves.items()):
            if nonconverging_flags[i] == 1:
                FD_Curves_copy[params] = dispForce
    else:
        for i, (params, dispForce) in enumerate(FD_Curves.items()):
            if nonconverging_flags[i] == 0:
                FD_Curves_copy[params] = dispForce

    return FD_Curves_copy    

def filter_simulations_simultaneous(combined_objective_value_to_param_FD_Curves, 
                                    objectives, 
                                    nonconverging_filter):
    
    objective_nonconverging_flags = {}
    for objective in objectives:
        objective_nonconverging_flags[objective] = check_convergence(combined_objective_value_to_param_FD_Curves[objective])
    
    if nonconverging_filter == True:
        indices_where_all_objectives_nonconverge = []
        for i in range(len(objective_nonconverging_flags[objectives[0]])):
            if all([objective_nonconverging_flags[objective][i] == 1 for objective in objectives]):
                indices_where_all_objectives_nonconverge.append(i)
    else:
        indices_where_all_objectives_converge = []
        for i in range(len(objective_nonconverging_flags[objectives[0]])):
            if all([objective_nonconverging_flags[objective][i] == 0 for objective in objectives]):
                indices_where_all_objectives_converge.append(i)

    combined_objective_value_to_param_FD_Curves_copy = {}
    for objective in objectives:
        combined_objective_value_to_param_FD_Curves_copy[objective] = {}
        for i, (params, dispForce) in enumerate(combined_objective_value_to_param_FD_Curves[objective].items()):
            if nonconverging_filter == True:
                if i in indices_where_all_objectives_nonconverge:
                    combined_objective_value_to_param_FD_Curves_copy[objective][params] = dispForce
            else:
                if i in indices_where_all_objectives_converge:
                    combined_objective_value_to_param_FD_Curves_copy[objective][params] = dispForce

    return combined_objective_value_to_param_FD_Curves_copy


def find_sim_center(dispForce):
    disp = dispForce["displacement"]
    force = dispForce["force"]
    # Find N largest values of force
    N = 20
    maxForce_N = np.partition(force, -N)[-N:]
    maxDispCorresponding_N = disp[np.argpartition(force, -N)[-N:]] 
    #maxForce = np.max(force)
    #maxDispCorresponding = disp[np.argmax(force)]
    maxForce = np.mean(maxForce_N)
    maxDispCorresponding = np.mean(maxDispCorresponding_N)
    simCenter = {"X": maxDispCorresponding, "Y": maxForce}
    return simCenter

def minmax_scaler(paramsTuple, paramConfig):
    scaledParams = []
    for paramName, paramValue in paramsTuple:
        paramMin = paramConfig[paramName]["iteration_lowerBound"]	
        paramMax = paramConfig[paramName]["iteration_upperBound"]
        scaledParam = (paramValue - paramMin) / (paramMax - paramMin)
        scaledParams.append(scaledParam)
    return scaledParams

def de_minmax_scaler(scaledParams, paramConfig):
    params = []
    for i, scaledParam in enumerate(scaledParams):
        paramName = list(paramConfig.keys())[i]
        paramMin = paramConfig[paramName]["iteration_lowerBound"]	
        paramMax = paramConfig[paramName]["iteration_upperBound"]
        param = scaledParam * (paramMax - paramMin) + paramMin
        params.append(param)
    return params