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

def reverseAsParamsToGeometries(curves, geometries):
    exampleGeometry = geometries[0]
    reverseCurves = {}
    for paramsTuple in curves[exampleGeometry]:
        reverseCurves[paramsTuple] = {}
        for geometry in geometries:
            reverseCurves[paramsTuple][geometry] = curves[geometry][paramsTuple]
    return reverseCurves

def calculate_yielding_index(targetDisplacement, targetForce, r2_threshold=0.998):
    """
    This function calculates the end of the elastic (linear) region of the force-displacement curve.
    """
    yielding_index = 0

    # Initialize the Linear Regression model
    linReg = LinearRegression()
    targetDisplacement = np.array(targetDisplacement)
    targetForce = np.array(targetForce)
    for i in range(2, len(targetDisplacement)):
        linReg.fit(targetDisplacement[:i].reshape(-1, 1), targetForce[:i]) 
        simForce = linReg.predict(targetDisplacement[:i].reshape(-1, 1)) 
        r2 = r2_score(targetForce[:i], simForce) 
        if r2 < r2_threshold:  # If R^2 is below threshold, mark the end of linear region
            yielding_index = i - 1
            break
    return yielding_index