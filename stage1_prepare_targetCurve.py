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
from stage0_configs import * 
from math import *
import json
from datetime import datetime
import os
import prettytable

def main_prepare_targetCurve(info):

    # ----------------------------------#
    #   Step 1: Preparing target curve  #
    # ----------------------------------#
    
    targetPath = info['targetPath']
    objectives = info['objectives']
    
    targetCurves = {}
    targetCenters = {}

    for objective in objectives:
        df = pd.read_excel(f"{targetPath}/{objective}/FD_Curve.xlsx", engine='openpyxl')
        expDisplacement = df['displacement/nm'].to_numpy()
        expForce = df['force/microN'].to_numpy()
        targetCurve = {}
        targetCurve['displacement'] = expDisplacement
        targetCurve['force'] = expForce
        # picking N points with largest force, and their corresponding displacement
        n = 50
        # get the indices of the n largest values
        indices = np.argpartition(expForce, -n)[-n:]
        # get the corresponding displacement
        largest_displacement = expDisplacement[indices]
        largest_force = expForce[indices]

        # Taking the mean is a good approximation for the center
        x_center, y_center = np.mean(largest_displacement), np.mean(largest_force)
        targetCenter = {"X": x_center, "Y": y_center}

        targetCurves[objective] = targetCurve
        targetCenters[objective] = targetCenter
        
    #print(targetCurves)
    #print(targetCenters)
    #time.sleep(180)
    return targetCurves, targetCenters

