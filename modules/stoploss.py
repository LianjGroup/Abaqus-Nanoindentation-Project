import numpy as np
import pandas as pd

from scipy.integrate import simpson
# import interp1d
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def lossFD(targetDisplacement, targetForce, simForce, iteration):
    pass

def stopFD_SOO(targetForce, simForce, yieldingIndex, deviationPercent):
    targetForceUpper = targetForce * (1 + 0.01 * deviationPercent)
    targetForceLower = targetForce * (1 - 0.01 * deviationPercent)
    return np.all((simForce[yieldingIndex:] >= targetForceLower[yieldingIndex:]) & (simForce[yieldingIndex:] <= targetForceUpper[yieldingIndex:]))

def stopFD(targetCurves, simCurves, geometries, yieldingIndices, deviationPercent):
    stopAllCurvesCheck = True
    for geometry in geometries:
        yieldingIndex = yieldingIndices[geometry]
        stopAllCurvesCheck = stopAllCurvesCheck & stopFD_SOO(targetCurves[geometry]['force'], simCurves[geometry]['force'], yieldingIndex, deviationPercent)
    return stopAllCurvesCheck