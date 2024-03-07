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
    
    ##############################
    # PARAMETER SAMPLING METHODS #
    ##############################
        
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

    ###############################
    # INITIAL SIMULATION PIPELINE #
    ###############################

    def run_initial_simulations(self, initialParams, currentIndices, batchNumber):
        deleteSimOutputs = self.info['deleteSimOutputs']
        indexParamsDict = self.create_indexParamsDict(initialParams, currentIndices)
        self.preprocess_simulations_initial(indexParamsDict)
        self.write_paths_initial(indexParamsDict)
        self.submit_array_jobs_initial(indexParamsDict)
        self.postprocess_results_initial(indexParamsDict, batchNumber)
        if deleteSimOutputs:
            self.delete_sim_outputs(indexParamsDict)

    ##############################
    # INITIAL SIMULATION METHODS #
    ##############################
                
    def create_indexParamsDict(self, initialParams, currentIndices):
        indexParamsDict = {}
        for order, paramDict in enumerate(initialParams):
            index = str(currentIndices[order])
            indexParamsDict[index] = tuple(paramDict.items())
        return indexParamsDict
    
    def preprocess_simulations_initial(self, indexParamsDict):
        simPath = self.info['simPath']
        templatePath = self.info['templatePath'] 
        CPLaw = self.info['CPLaw']
        
        for index, paramsTuple in indexParamsDict.items():
            # Create the simulation folder if not exists, else delete the folder and create a new one
            if os.path.exists(f"{simPath}/initial/{index}"):
                shutil.rmtree(f"{simPath}/initial/{index}")
            shutil.copytree(templatePath, f"{simPath}/initial/{index}")
            replace_parameters_geometry_inp(f"{simPath}/initial/{index}/geometry.inp", dict(paramsTuple), CPLaw)
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
    
    def postprocess_results_initial(self, indexParamsDict, batchNumber):

        simPath = self.info['simPath']
        resultPath = self.info['resultPath']
        logPath = self.info['logPath']
        
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
                    
        # Returning force-displacement curve data
        if batchNumber == "all":
            np.save(f"{resultPath}/initial/common/FD_Curves.npy", FD_Curves)
            printLog("Saving successfully FD_Curves.npy results", logPath)
        else:
            np.save(f"{resultPath}/initial/common/FD_Curves_batch_{batchNumber}.npy", FD_Curves)
            printLog(f"Saving successfully FD_Curves_batch_{batchNumber}.npy results", logPath)
    
    def delete_sim_outputs(self, indexParamsDict):
        simPath = self.info['simPath']
        for index, paramsTuple in indexParamsDict.items():
            shutil.rmtree(f"{simPath}/initial/{index}")

    #################################
    # ITERATION SIMULATION PIPELINE #
    #################################
    
    def run_iteration_simulations(self, paramsDict, iterationIndex):
        self.preprocess_simulations_iteration(paramsDict, iterationIndex)
        self.write_paths_iteration(iterationIndex)
        self.submit_array_jobs_iteration()
        grainSR_to_params_FD_Curves = self.postprocess_results_iteration(paramsDict, iterationIndex)
        return grainSR_to_params_FD_Curves

    ################################
    # ITERATION SIMULATION METHODS #
    ################################

    def preprocess_simulations_iteration(self, paramsDict, iterationIndex):
        simPath = self.info['simPath']
        templatePath = self.info['templatePath'] 
        objectives = self.info['objectives']
        CPLaw = self.info['CPLaw']
        
        paramsTuple = tuple(paramsDict.items())

        # Create the simulation folder if not exists, else delete the folder and create a new one
        for objective in objectives:
            if os.path.exists(f"{simPath}/{objective}/iteration/{iterationIndex}"):
                shutil.rmtree(f"{simPath}/{objective}/iteration/{iterationIndex}")
            shutil.copytree(f"{templatePath}/{objective}", f"{simPath}/{objective}/iteration/{iterationIndex}")
            replace_parameters_geometry_inp(f"{simPath}/{objective}/iteration/{iterationIndex}/geometry.inp", paramsDict, CPLaw)
            create_parameter_file(f"{simPath}/{objective}/iteration/{iterationIndex}", dict(paramsTuple))        


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
