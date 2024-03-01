import os
import pandas as pd

#########################################################
# Creating necessary directories for the configurations #
#########################################################

def checkCreate(path):
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_template():

    # For configs
    path = "configs"
    checkCreate(path)
    # Create an empty global_config.xlsx file 
    df = pd.DataFrame()
    df.to_excel(f"{path}/global_config.xlsx", index=False)

    # For linux_slurm
    path = "linux_slurm"
    checkCreate(path)

    # for modules
    path = "modules"
    checkCreate(path)

    # for notebooks
    path = "notebooks"
    checkCreate(path)

    # for optimizers
    path = "optimizers"
    checkCreate(path)
    
if __name__ == "__main__":    
    initialize_template()
    

    