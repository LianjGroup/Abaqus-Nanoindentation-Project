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

def filter_simulations(FD_Curves, nonconverging):
    # Converting the unit
    maxDispCorrespondingToMaxForce = -np.inf

    for params, dispForce in FD_Curves.items():
        disp = dispForce["displacement"]
        force = dispForce["force"]
        if maxDispCorrespondingToMaxForce < disp[np.argmax(force)]: 
            maxDispCorrespondingToMaxForce = disp[np.argmax(force)]

    # print(f"Max disp corresponding to max force: {maxDispCorrespondingToMaxForce}")
    
    FD_Curves_copy = copy.deepcopy(FD_Curves)
    
    for i, (params, dispForce) in enumerate(FD_Curves.items()):  
        disp = dispForce["displacement"]
        force = dispForce["force"]
        dispCorrespondingToMaxForce = disp[np.argmax(force)]

        if nonconverging == True:
            if not dispCorrespondingToMaxForce < maxDispCorrespondingToMaxForce - 2:
                del FD_Curves_copy[params]
                #print(f"Disp corresponding to max force: {disp[np.argmax(force)]} (converging)")
            #else:
                #print(f"Disp corresponding to max force: {disp[np.argmax(force)]} (nonconverging)")
        else:
            if dispCorrespondingToMaxForce < maxDispCorrespondingToMaxForce - 2:
                #print(f"Disp corresponding to max force: {disp[np.argmax(force)]} (nonconverging)")
                del FD_Curves_copy[params] 
            #else:
                #print(f"Disp corresponding to max force: {disp[np.argmax(force)]} (converging)")
    return FD_Curves_copy    

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