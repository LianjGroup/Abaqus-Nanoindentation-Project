import numpy as np
import pandas as pd

from scipy.integrate import simpson
# import interp1d
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from modules.calculation import *

def lossFD_SOO(targetCenter, simCenter):
    targetCenterX = targetCenter['X']
    targetCenterY = targetCenter['Y']
    simCenterX = simCenter['X']
    simCenterY = simCenter['Y']
    return np.sqrt((targetCenterX - simCenterX)**2 + (targetCenterY - simCenterY)**2)

def stopFD_SOO(targetCenter, simCenter, deviationPercent):
    targetCenterX = targetCenter['X']
    targetCenterY = targetCenter['Y']
    simCenterX = simCenter['X']
    simCenterY = simCenter['Y']
    deviationPercentX = deviationPercent['X']
    deviationPercentY = deviationPercent['Y']

    if abs(targetCenterX - simCenterX) / targetCenterX < 0.01 * deviationPercentX:
        if abs(targetCenterY - simCenterY) / targetCenterY < 0.01 * deviationPercentY:
            return True
    return False

def stopFD_MOO(targetCenters, simCenters, deviationPercent, objectives):
    stopAllObjectivesCheckObjectives = {}

    for objective in objectives:
        stopObjectiveCheck = stopFD_SOO(targetCenters[objective], simCenters[objective], deviationPercent)
        stopAllObjectivesCheckObjectives[objective] = stopObjectiveCheck
    
    stopAllObjectives = all(stopAllObjectivesCheckObjectives.values())
    return stopAllObjectives, stopAllObjectivesCheckObjectives



    