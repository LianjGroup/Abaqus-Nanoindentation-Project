import os
import pandas as pd

#########################################################
# Creating necessary directories for the configurations #
#########################################################

def checkCreate(path):
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_directory(material, CPLaw, grains, strainRates):

    # optimizing instance
    optimizing_instance = f"{CPLaw}_{material}"

    # list of objectives 
    objectives = []
    for grain in grains:
        for strainRate in strainRates:
            objectives.append(f"grain_{grain}_sr_{strainRate}")
    
    # For paramInfo
    path = f"paramInfo/{optimizing_instance}"
    checkCreate(path)
    
    # For results 
    path = f"results/{optimizing_instance}"
    checkCreate(path)
    for objective in objectives:
        checkCreate(f"{path}/{objective}")
        checkCreate(f"{path}/{objective}/initial")
        checkCreate(f"{path}/{objective}/initial/data")
        checkCreate(f"{path}/{objective}/initial/common")
        checkCreate(f"{path}/{objective}/iteration")
        checkCreate(f"{path}/{objective}/iteration/data")
        checkCreate(f"{path}/{objective}/iteration/common")

    # For simulations
    path = f"simulations/{optimizing_instance}"
    checkCreate(path)
    for grain in grains:
        for strainRate in strainRates:
            checkCreate(f"{path}/{objective}")
            checkCreate(f"{path}/{objective}/initial")
            checkCreate(f"{path}/{objective}/iteration")

    # For targets
    path = f"targets/{optimizing_instance}"
    checkCreate(path)
    for objective in objectives:
        checkCreate(f"{path}/{objective}")

    # For templates
    path = f"templates/{optimizing_instance}"
    checkCreate(path)
    for grain in grains:
        for strainRate in strainRates:
            checkCreate(f"{path}/{objective}")

    # The project path folder
    projectPath = os.getcwd()
    
    # The logging path
    logPath = f"log/{optimizing_instance}.txt"
    # The paramInfo path
    paramInfoPath = f"paramInfo/{optimizing_instance}"
    # The results path
    resultPath = f"results/{optimizing_instance}"
    # The simulations path
    simPath = f"simulations/{optimizing_instance}"
    # The target path
    targetPath = f"targets/{optimizing_instance}"
    # The templates path
    templatePath = f"templates/{optimizing_instance}"

    return optimizing_instance, objectives, projectPath, logPath, paramInfoPath, resultPath, simPath, templatePath, targetPath

if __name__ == "__main__":
    globalConfig = pd.read_excel("configs/global_config.xlsx", nrows=1, engine="openpyxl")
    globalConfig = globalConfig.T.to_dict()[0]
    material = globalConfig["material"]
    CPLaw = globalConfig["CPLaw"]
    grains = globalConfig["grains"]
    strainRates = globalConfig["strainRates"]

    grains = grains.split(";")
    strainRates = strainRates.split(";")
    
    initialize_directory(material, CPLaw, grains, strainRates)
    

    