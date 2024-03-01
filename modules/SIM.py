import pandas as pd
import numpy as np
import subprocess
import os
from modules.IO import *
from modules.calculation import *
from optimizers.optimize import *
import sys
import shutil
import random
import time
import sobol_seq

class SIMULATION():
    def __init__(self, info):
        self.info = info
   
    def latin_hypercube_sampling(self):
        paramConfig = self.info["paramConfig"]
        numberOfInitialSims = self.info["numberOfInitialSims"]
        linspaceValues = {}
        for param in paramConfig:
            linspaceValues[param] = np.linspace(
                start=paramConfig[param]["initial_lowerBound"] * paramConfig[param]["exponent"], 
                stop=paramConfig[param]["initial_upperBound"] * paramConfig[param]["exponent"], 
                num = self.info["initialSimsSpacing"])
            linspaceValues[param] = linspaceValues[param].tolist()   
        points = []
        for _ in range(numberOfInitialSims):
            while True:
                candidateParam = {}
                for param in linspaceValues:
                    random.shuffle(linspaceValues[param])
                    candidateParam[param] = linspaceValues[param].pop()
                if candidateParam not in points:
                    break
            points.append(candidateParam)
        return points
    
    def sobol_sequence_sampling(self):
        paramConfig = self.info["paramConfig"]
        numberOfInitialSims = self.info["numberOfInitialSims"]
        num_params = len(paramConfig)
        # Generate Sobol sequence samples
        sobol_samples = sobol_seq.i4_sobol_generate(num_params, numberOfInitialSims)
        # Scale samples to parameter ranges
        points = []
        for sample in sobol_samples:
            scaled_sample = {}
            for i, param in enumerate(paramConfig):
                lower_bound = paramConfig[param]["initial_lowerBound"] * paramConfig[param]["exponent"]
                upper_bound = paramConfig[param]["initial_upperBound"] * paramConfig[param]["exponent"]
                # Scale the Sobol sample for this parameter
                scaled_sample[param] = lower_bound + (upper_bound - lower_bound) * sample[i]
            points.append(scaled_sample)
        return points

    def run_initial_simulations(self, initialParams):
        maxConcurrentSimNumber = self.info['maxConcurrentSimNumber']
        if maxConcurrentSimNumber == "max" or maxConcurrentSimNumber >= len(initialParams):
            indexParamsDict = self.create_indexParamsDict(initialParams)
            self.preprocess_simulations_initial(indexParamsDict)
            self.write_paths_initial(indexParamsDict)
            self.submit_array_jobs_initial(indexParamsDict)
            self.postprocess_results_initial(indexParamsDict)
        else:
            indexParamsDict = self.create_indexParamsDict(initialParams)
            for i in range(0, len(initialParams), maxConcurrentSimNumber):
                if i + maxConcurrentSimNumber > len(initialParams):
                    sub_indexParamsDict = dict(list(indexParamsDict.items())[i:])
                    sub_initialParams = initialParams[i:]
                else:
                    sub_indexParamsDict = dict(list(indexParamsDict.items())[i:i+maxConcurrentSimNumber])
                    sub_initialParams = initialParams[i:i+maxConcurrentSimNumber]
                # print(sub_indexParamsDict)
                # print(sub_initialParams)
                # time.sleep(180)
                indexParamsDict = self.preprocess_simulations_initial(sub_initialParams, sub_indexParamsDict)
                self.write_paths_initial()
                self.submit_array_jobs_initial()
                self.postprocess_results_initial(sub_indexParamsDict)

    def create_indexParamsDict(self, initialParams):
        indexParamsDict = {}
        for index, paramDict in enumerate(initialParams):
            indexParamsDict[str(index+1)] = tuple(paramDict.items())
        return indexParamsDict
    
    def preprocess_simulations_initial(self, indexParamsDict):
        simPath = self.info['simPath']
        templatePath = self.info['templatePath'] 
        
        for index, paramsTuple in indexParamsDict.items():
            # Create the simulation folder if not exists, else delete the folder and create a new one
            if os.path.exists(f"{simPath}/initial/{index}"):
                shutil.rmtree(f"{simPath}/initial/{index}")
            shutil.copytree(templatePath, f"{simPath}/initial/{index}")
            replace_parameters_geometry_inp(f"{simPath}/initial/{index}/geometry.inp", dict(paramsTuple))
            create_parameter_file(f"{simPath}/initial/{index}", dict(paramsTuple))

        return indexParamsDict

    def write_paths_initial(self, indexParamsDict):
        projectPath = self.info['projectPath']
        simPath = self.info['simPath']

        with open("linux_slurm/array_file.txt", 'w') as filename:
            for index in list(indexParamsDict.keys()):
                filename.write(f"{projectPath}/{simPath}/initial/{index}\n")
    
    def submit_array_jobs_initial(self, indexParamsDict):
        logPath = self.info['logPath']        
        indices = ",".join(list(indexParamsDict.keys()))

        printLog("Initial simulation preprocessing stage starts", logPath)
        printLog(f"Number of jobs required: {len(indexParamsDict)}", logPath)
        subprocess.run(f"sbatch --wait --array={indices} linux_slurm/puhti_abaqus_array_small.sh", shell=True)
        printLog("Initial simulation postprocessing stage finished", logPath)
    
    def postprocess_results_initial(self, indexParamsDict):
        numberOfInitialSims = self.info['numberOfInitialSims']
        simPath = self.info['simPath']
        resultPath = self.info['resultPath']
        logPath = self.info['logPath']
        deleteOutputSims = self.info['deleteOutputSims']
        # The structure of force-displacement curve: dict of (CP params tuple of tuples) -> {force: forceArray , displacement: displacementArray}

        FD_Curves = {}
        for index, paramsTuple in indexParamsDict.items():
            if not os.path.exists(f"{resultPath}/initial/data/{index}"):
                os.mkdir(f"{resultPath}/initial/data/{index}")
            shutil.copy(f"{simPath}/initial/{index}/FD_Curve.txt", f"{resultPath}/initial/data/{index}")
            shutil.copy(f"{simPath}/initial/{index}/parameters.xlsx", f"{resultPath}/initial/data/{index}")
            shutil.copy(f"{simPath}/initial/{index}/parameters.csv", f"{resultPath}/initial/data/{index}")
                        
            displacement, force = read_FD_Curve(f"{simPath}/initial/{index}/FD_Curve.txt")
            create_FD_Curve_file(f"{resultPath}/initial/data/{index}", displacement, force)

            FD_Curves[paramsTuple] = {}
            FD_Curves[paramsTuple]['displacement'] = displacement
            FD_Curves[paramsTuple]['force'] = force
            
        if deleteOutputSims == "yes":
            for index, paramsTuple in indexParamsDict.items():
                shutil.rmtree(f"{simPath}/initial/{index}")
        # Returning force-displacement curve data
        np.save(f"{resultPath}/initial/common/FD_Curves.npy", FD_Curves)
        printLog("Saving successfully simulation results", logPath)

    def run_iteration_simulations(self, paramsDict, iterationIndex):
        self.preprocess_simulations_iteration(paramsDict, iterationIndex)
        self.write_paths_iteration(iterationIndex)
        #time.sleep(180)
        self.submit_array_jobs_iteration()
        grainSR_to_params_FD_Curves = self.postprocess_results_iteration(paramsDict, iterationIndex)
        return grainSR_to_params_FD_Curves, geom_to_params_flowCurves
    
    def preprocess_simulations_iteration(self, paramsDict, iterationIndex):
        resultPath = self.info['resultPath']
        simPath = self.info['simPath']
        geometries = self.info['geometries']
        templatePath = self.info['templatePath'] 
        hardeningLaw = self.info['hardeningLaw']
        numberOfInitialSims = self.info['numberOfInitialSims']
        truePlasticStrain = self.info['truePlasticStrain']
        maxTargetDisplacements = self.info['maxTargetDisplacements']
        
        paramsTuple = tuple(paramsDict.items())
        trueStress = calculate_flowCurve(paramsDict, hardeningLaw, truePlasticStrain)
        geom_to_params_flowCurves = {}

        for geometry in geometries:
            geom_to_params_flowCurves[geometry] = {}
            geom_to_params_flowCurves[geometry][paramsTuple] = {}
            geom_to_params_flowCurves[geometry][paramsTuple]['strain'] = truePlasticStrain
            geom_to_params_flowCurves[geometry][paramsTuple]['stress'] = trueStress
        
        # Create the simulation folder if not exists, else delete the folder and create a new one
        for geometry in geometries:
            if os.path.exists(f"{simPath}/{geometry}/iteration/{iterationIndex}"):
                shutil.rmtree(f"{simPath}/{geometry}/iteration/{iterationIndex}")
            shutil.copytree(f"{templatePath}/{geometry}", f"{simPath}/{geometry}/iteration/{iterationIndex}")
            truePlasticStrain = geom_to_params_flowCurves[geometry][paramsTuple]['strain']
            trueStress = geom_to_params_flowCurves[geometry][paramsTuple]['stress']
            replace_flowCurve_material_inp(f"{simPath}/{geometry}/iteration/{iterationIndex}/material.inp", truePlasticStrain, trueStress)
            replace_maxDisp_geometry_inp(f"{simPath}/{geometry}/iteration/{iterationIndex}/geometry.inp", maxTargetDisplacements[geometry])
            replace_materialName_geometry_inp(f"{simPath}/{geometry}/iteration/{iterationIndex}/geometry.inp", "material.inp")
            create_parameter_file(f"{simPath}/{geometry}/iteration/{iterationIndex}", dict(paramsTuple))
            create_flowCurve_file(f"{simPath}/{geometry}/iteration/{iterationIndex}", truePlasticStrain, trueStress)
        
        return geom_to_params_flowCurves

    def write_paths_iteration(self, iterationIndex):
        projectPath = self.info['projectPath']
        simPath = self.info['simPath']
        geometries = self.info['geometries']
        with open("linux_slurm/array_file.txt", 'w') as filename:
            for geometry in geometries:
                filename.write(f"{projectPath}/{simPath}/{geometry}/iteration/{iterationIndex}\n")

    def submit_array_jobs_iteration(self):
        logPath = self.info['logPath']       

        geometries = self.info['geometries']
        printLog("Iteration simulation preprocessing stage starts", logPath)
        printLog(f"Number of jobs required: {len(geometries)}", logPath)
        subprocess.run(f"sbatch --wait --array=1-{len(geometries)} linux_slurm/puhti_abaqus_array_small.sh", shell=True)
        printLog("Iteration simulation postprocessing stage finished", logPath)

    def postprocess_results_iteration(self, paramsDict, iterationIndex):
        simPath = self.info['simPath']
        resultPath = self.info['resultPath']
        logPath = self.info['logPath']
        geometries = self.info['geometries']

        paramsTuple = tuple(paramsDict.items())
        geom_to_params_FD_Curves = {}
        
        
        for geometry in geometries:
            if not os.path.exists(f"{resultPath}/{geometry}/iteration/data/{iterationIndex}"):
                os.mkdir(f"{resultPath}/{geometry}/iteration/data/{iterationIndex}")
            shutil.copy(f"{simPath}/{geometry}/iteration/{iterationIndex}/FD_Curve.txt", f"{resultPath}/{geometry}/iteration/data/{iterationIndex}")
            shutil.copy(f"{simPath}/{geometry}/iteration/{iterationIndex}/parameters.xlsx", f"{resultPath}/{geometry}/iteration/data/{iterationIndex}")
            shutil.copy(f"{simPath}/{geometry}/iteration/{iterationIndex}/parameters.csv", f"{resultPath}/{geometry}/iteration/data/{iterationIndex}")
                        
            displacement, force = read_FD_Curve(f"{simPath}/{geometry}/iteration/{iterationIndex}/FD_Curve.txt")
            
            geom_to_params_FD_Curves[geometry] = {}
            geom_to_params_FD_Curves[geometry][paramsTuple] = {}
            geom_to_params_FD_Curves[geometry][paramsTuple]['displacement'] = displacement
            geom_to_params_FD_Curves[geometry][paramsTuple]['force'] = force
            create_FD_Curve_file(f"{resultPath}/{geometry}/iteration/data/{iterationIndex}", displacement, force)
        printLog("Saving successfully iteration simulation results", logPath)
        return geom_to_params_FD_Curves
