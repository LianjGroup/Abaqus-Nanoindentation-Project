#######################################
# Simulation related helper functions #
#######################################

import numpy as np
import pandas as pd
from prettytable import PrettyTable

def printLog(message, logPath):
    with open(logPath, 'a+') as logFile:
        logFile.writelines(message + "\n")
    print(message)

def prettyPrint(parameters, paramConfig, logPath):
    logTable = PrettyTable()
    logTable.field_names = ["Parameter", "Value"]
    for param in parameters:
        paramName = paramConfig[param]['name']
        paramValue = parameters[param]
        paramUnit = paramConfig[param]['unit']
        paramValueUnit = f"{paramValue} {paramUnit}" if paramUnit != "dimensionless" else paramValue
        #print(paramName)
        #print(paramValueUnit)
        logTable.add_row([paramName, paramValueUnit])

    stringMessage = "\n"
    stringMessage += logTable.get_string()
    stringMessage += "\n"

    printLog(stringMessage, logPath)

def read_FD_Curve(filePath):
    output_data=np.loadtxt(filePath, skiprows=2)
    # column 1 is time step
    # column 2 is displacement
    # column 3 is force
    columns=['X', 'Displacement', 'Force']
    df = pd.DataFrame(data=output_data, columns=columns)
    # Converting to numpy array
    displacement = df.iloc[:, 1].to_numpy()
    force = df.iloc[:, 2].to_numpy()
    return displacement, force

def create_parameters_file(filePath, paramsDict):
    columns = ["Parameter", "Value"]
    df = pd.DataFrame(columns=columns)
    for key, value in paramsDict.items():
        df.loc[len(df.index)] = [key, value]
    df.to_excel(f"{filePath}/parameters.xlsx", index=False)
    df.to_csv(f"{filePath}/parameters.csv", index=False)


def create_FD_Curve_file(filePath, displacement, force):
    columns = ["displacement,mm", "force,kN", "force,N"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(displacement)):
        df.loc[len(df.index)] = [displacement[i], force[i] * 1e-3, force[i]]
    df.to_excel(f"{filePath}/FD_Curve.xlsx", index=False)
    df.to_csv(f"{filePath}/FD_Curve.csv", index=False)

def replace_parameters_into_inp(filePath, paramsDict, CPLaw):
    if CPLaw == 'PH':
        with open(filePath, 'r') as geometry_inp:
            geometry_inp_content = geometry_inp.readlines()
        start_line = None
        end_line = None
        # Replacing tau0 value
        for i, line in enumerate(geometry_inp_content[-500:]):
            if line.startswith('*USER MATERIAL,CONSTANTS=23,UNSYMM'):
                line_containing_tau0 = geometry_inp_content[-500 + i + 1]
                line_containing_tau0_split = line_containing_tau0.split(',')
                line_containing_tau0_split[3] = str(paramsDict['tau0'])
                line_containing_tau0_new = ','.join(line_containing_tau0_split)
                geometry_inp_content[-500 + i + 1] = line_containing_tau0_new
                break
        
        # Replacing a, h0, tausat values
        for i, line in enumerate(geometry_inp_content[-500:]):
            if line.startswith('** Q , 2 VECTORS, IHARDMODEL,'):
                line_containing_others = geometry_inp_content[-500 + i + 1]
                line_containing_others_split = line_containing_others.split(',')
                line_containing_others_split[2] = str(paramsDict['a'])
                line_containing_others_split[3] = str(paramsDict['tausat'])
                line_containing_others_split[4] = str(paramsDict['h0'])
                line_containing_others_new = ','.join(line_containing_others_split)
                geometry_inp_content[-500 + i + 1] = line_containing_others_new
                break

        with open(filePath, 'w') as file:
            file.writelines(geometry_inp_content)
    elif CPLaw == 'DB':
        pass