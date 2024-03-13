from modules.IO import *
from modules.calculation import *
from optimizers.optimize import *
from modules.stoploss import *
import stage0_configs 
import stage1_prepare_targetCurve
import stage2_run_initialSims
import stage3_prepare_simCurves
import stage4_iterative_calibration
from math import *
import os

def main_pipeline():

    # -------------------------------#
    #  Automated optimization stages #
    # -------------------------------#

    info = stage0_configs.main_config()
    
    logPath = info['logPath']
    deviationPercent = info['deviationPercent']

    targetCurves, targetCenters = stage1_prepare_targetCurve.main_prepare_targetCurve(info)
    info['targetCurves'] = targetCurves
    info['targetCenters'] = targetCenters

    stage2_run_initialSims.main_run_initialSims(info)
    
    FD_Curves_dict = stage3_prepare_simCurves.main_prepare_simCurves(info) 
    info["FD_Curves_dict"] = FD_Curves_dict
    
    stage4_iterative_calibration.main_iterative_calibration(info)

    printLog(f"The simulations have satisfied the {deviationPercent}% deviation stop condition", logPath)
    printLog("Parameter calibration has successfully completed", logPath)
    
if __name__ == "__main__":
    main_pipeline()