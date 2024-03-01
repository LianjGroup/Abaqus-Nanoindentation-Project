import os
import pandas as pd

#########################################################
# Creating necessary directories for the configurations #
#########################################################

def checkCreate(path):
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_directory(material, hardeningLaw, geometries, curveIndex):

    # For log
    checkCreate("log")

    # For paramInfo
    path = f"paramInfo/{material}_{hardeningLaw}_curve{curveIndex}"
    checkCreate(path)
    
    # For results 
    path = f"results/{material}_{hardeningLaw}_curve{curveIndex}"
    checkCreate(path)
    for geometry in geometries:
        checkCreate(f"{path}/{geometry}")
        checkCreate(f"{path}/{geometry}/initial")
        checkCreate(f"{path}/{geometry}/initial/data")
        checkCreate(f"{path}/{geometry}/initial/common")
        checkCreate(f"{path}/{geometry}/iteration")
        checkCreate(f"{path}/{geometry}/iteration/data")
        checkCreate(f"{path}/{geometry}/iteration/common")

    # For simulations
    path = f"simulations/{material}_{hardeningLaw}_curve{curveIndex}"
    checkCreate(path)
    for geometry in geometries:
        checkCreate(f"{path}/{geometry}")
        checkCreate(f"{path}/{geometry}/initial")
        checkCreate(f"{path}/{geometry}/iteration")

    # For targets
    path = f"targets/{material}_{hardeningLaw}_curve{curveIndex}"
    checkCreate(path)
    for geometry in geometries:
        checkCreate(f"{path}/{geometry}")

    # For templates
    path = f"templates/{material}"
    checkCreate(path)
    for geometry in geometries:
        checkCreate(f"{path}/{geometry}")

    # The project path folder
    projectPath = os.getcwd()
    
    # The logging path
    logPath = f"log/{material}_{hardeningLaw}_curve{curveIndex}.txt"
    # The paramInfo path
    paramInfoPath = f"paramInfo/{material}_{hardeningLaw}_curve{curveIndex}"
    # The results path
    resultPath = f"results/{material}_{hardeningLaw}_curve{curveIndex}"
    # The simulations path
    simPath = f"simulations/{material}_{hardeningLaw}_curve{curveIndex}"
    # The target path
    targetPath = f"targets/{material}_{hardeningLaw}_curve{curveIndex}"
    # The templates path
    templatePath = f"templates/{material}"

    return projectPath, logPath, paramInfoPath, resultPath, simPath, templatePath, targetPath

if __name__ == "__main__":
    globalConfig = pd.read_excel("configs/global_config.xlsx", nrows=1, engine="openpyxl")
    globalConfig = globalConfig.T.to_dict()[0]
    material = globalConfig["material"]
    optimizerName = globalConfig["optimizerName"]
    hardeningLaw = globalConfig["hardeningLaw"]
    deviationPercent = globalConfig["deviationPercent"]
    geometry = globalConfig["geometry"]
    curveIndex = globalConfig["curveIndex"]
    numberOfInitialSims = globalConfig["numberOfInitialSims"]
    initialSimsSpacing = globalConfig["initialSimsSpacing"]

    geometries = geometry.split(",")
    initialize_directory(material, hardeningLaw, geometries, curveIndex)
    

    