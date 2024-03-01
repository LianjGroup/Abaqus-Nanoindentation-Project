import os
import pandas as pd

#########################################################
# Creating necessary directories for the configurations #
#########################################################

def checkCreate(path):
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_directory(material, CPLaw, grains, strainRates):

    # For paramInfo
    path = f"paramInfo/{CPLaw}_{material}"
    checkCreate(path)
    
    # For results 
    path = f"results/{CPLaw}_{material}"
    checkCreate(path)
    for grain in grains:
        for strainRate in strainRates:
            checkCreate(f"{path}/grain_{grain}_sr_{strainRate}")
            checkCreate(f"{path}/grain_{grain}_sr_{strainRate}/initial")
            checkCreate(f"{path}/grain_{grain}_sr_{strainRate}/initial/data")
            checkCreate(f"{path}/grain_{grain}_sr_{strainRate}/initial/common")
            checkCreate(f"{path}/grain_{grain}_sr_{strainRate}/iteration")
            checkCreate(f"{path}/grain_{grain}_sr_{strainRate}/iteration/data")
            checkCreate(f"{path}/grain_{grain}_sr_{strainRate}/iteration/common")

    # For simulations
    path = f"simulations/{CPLaw}_{material}"
    checkCreate(path)
    for grain in grains:
        for strainRate in strainRates:
            checkCreate(f"{path}/grain_{grain}_sr_{strainRate}")
            checkCreate(f"{path}/grain_{grain}_sr_{strainRate}/initial")
            checkCreate(f"{path}/grain_{grain}_sr_{strainRate}/iteration")

    # For targets
    path = f"targets/{CPLaw}_{material}"
    checkCreate(path)
    for grain in grains:
        for strainRate in strainRates:
            checkCreate(f"{path}/grain_{grain}_sr_{strainRate}")

    # For templates
    path = f"templates/{CPLaw}_{material}"
    checkCreate(path)
    for grain in grains:
        for strainRate in strainRates:
            checkCreate(f"{path}/grain_{grain}_sr_{strainRate}")

    # The project path folder
    projectPath = os.getcwd()
    
    # The logging path
    logPath = f"log/{CPLaw}_{material}.txt"
    # The paramInfo path
    paramInfoPath = f"paramInfo/{CPLaw}_{material}"
    # The results path
    resultPath = f"results/{CPLaw}_{material}"
    # The simulations path
    simPath = f"simulations/{CPLaw}_{material}"
    # The target path
    targetPath = f"targets/{CPLaw}_{material}"
    # The templates path
    templatePath = f"templates/{CPLaw}_{material}"

    return projectPath, logPath, paramInfoPath, resultPath, simPath, templatePath, targetPath

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
    

    