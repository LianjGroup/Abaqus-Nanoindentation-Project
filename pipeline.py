import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from modules.hardeningLaws import *
from modules.IO import *
from modules.calculation import *
from optimizers.optimize import *
from modules.stoploss import *
from optimizers.BO import *
import stage0_configs 
import stage1_prepare_targetCurve
import stage2_run_initialSims
import stage3_prepare_simCurves
import stage4_iterative_calibration
from math import *
import json
from datetime import datetime
import os
import prettytable

def main_pipeline():

    # -------------------------------#
    #  Automated optimization stages #
    # -------------------------------#

    info = stage0_configs.main_config()
    
    projectPath = info['projectPath']
    logPath = info['logPath']
    resultPath = info['resultPath']
    simPath = info['simPath']
    targetPath = info['targetPath']
    templatePath = info['templatePath'] 
    material = info['material']
    optimizerName = info['optimizerName']
    hardeningLaw = info['hardeningLaw']
    paramConfig = info['paramConfig']
    deviationPercent = info['deviationPercent']
    numberOfInitialSims = info['numberOfInitialSims']

    targetCurves, maxTargetDisplacements = stage1_prepare_targetCurve.main_prepare_targetCurve(info)
    info['targetCurves'] = targetCurves
    info['maxTargetDisplacements'] = maxTargetDisplacements

    stage2_run_initialSims.main_run_initialSims(info)
    
    FD_Curves_dict, flowCurves_dict = stage3_prepare_simCurves.main_prepare_simCurves(info) 
    info["initial_original_geom_to_param_FD_Curves_smooth"] = FD_Curves_dict['initial_original_geom_to_param_FD_Curves_smooth']
    info["iteration_original_geom_to_param_FD_Curves_smooth"] = FD_Curves_dict['iteration_original_geom_to_param_FD_Curves_smooth']
    info["combined_original_geom_to_param_FD_Curves_smooth"] = FD_Curves_dict['combined_original_geom_to_param_FD_Curves_smooth']
    info["initial_interpolated_geom_to_param_FD_Curves_smooth"] = FD_Curves_dict['initial_interpolated_geom_to_param_FD_Curves_smooth']
    info["iteration_interpolated_geom_to_param_FD_Curves_smooth"] = FD_Curves_dict['iteration_interpolated_geom_to_param_FD_Curves_smooth']
    info["combined_interpolated_geom_to_param_FD_Curves_smooth"] = FD_Curves_dict['combined_interpolated_geom_to_param_FD_Curves_smooth']
    info['iteration_original_geom_to_param_FD_Curves_unsmooth'] = FD_Curves_dict['iteration_original_geom_to_param_FD_Curves_unsmooth'] 
    info['combined_interpolated_param_to_geom_FD_Curves_smooth'] = FD_Curves_dict['combined_interpolated_param_to_geom_FD_Curves_smooth'] 
    info['iteration_original_param_to_geom_FD_Curves_smooth'] = FD_Curves_dict['iteration_original_param_to_geom_FD_Curves_smooth']
    info["initial_original_geom_to_param_flowCurves"] = flowCurves_dict['initial_original_geom_to_param_flowCurves']
    info["iteration_original_geom_to_param_flowCurves"] = flowCurves_dict['iteration_original_geom_to_param_flowCurves']
    info["combined_original_geom_to_param_flowCurves"] = flowCurves_dict['combined_original_geom_to_param_flowCurves']
    
    stage4_iterative_calibration.main_iterative_calibration(info)

    printLog(f"The simulations have satisfied the {deviationPercent}% deviation stop condition")
    printLog("Parameter calibration has successfully completed", logPath)
    

if __name__ == "__main__":
    main_pipeline()