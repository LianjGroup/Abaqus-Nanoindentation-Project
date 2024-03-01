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

def create_parameter_file(filePath, paramsDict):
    columns = ["Parameter", "Value"]
    df = pd.DataFrame(columns=columns)
    for key, value in paramsDict.items():
        df.loc[len(df.index)] = [key, value]
    df.to_excel(f"{filePath}/parameters.xlsx", index=False)
    df.to_csv(f"{filePath}/parameters.csv", index=False)

def create_flowCurve_file(filePath, truePlasticStrain, trueStress):
    columns = ["strain,-", "stress,MPa", "stress,Pa"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(truePlasticStrain)):
        df.loc[len(df.index)] = [truePlasticStrain[i], trueStress[i], trueStress[i]*1e6]
    df.to_excel(f"{filePath}/flowCurve.xlsx", index=False)
    df.to_csv(f"{filePath}/flowCurve.csv", index=False)

def create_FD_Curve_file(filePath, displacement, force):
    columns = ["displacement,mm", "force,kN", "force,N"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(displacement)):
        df.loc[len(df.index)] = [displacement[i], force[i] * 1e-3, force[i]]
    df.to_excel(f"{filePath}/FD_Curve.xlsx", index=False)
    df.to_csv(f"{filePath}/FD_Curve.csv", index=False)

def replace_flowCurve_material_inp(filePath, truePlasticStrain, trueStress):
    with open(filePath, 'r') as material_inp:
        material_inp_content = material_inp.readlines()
    # Locate the section containing the stress-strain data
    start_line = None
    end_line = None
    for i, line in enumerate(material_inp_content):
        if '*Plastic' in line:
            start_line = i + 1
        elif '*Density' in line:
            end_line = i
            break

    if start_line is None or end_line is None:
        raise ValueError('Could not find the stress-strain data section')

    # Modify the stress-strain data
    new_stress_strain_data = zip(trueStress, truePlasticStrain)
    # Update the .inp file
    new_lines = []
    new_lines.extend(material_inp_content[:start_line])
    new_lines.extend([f'{stress},{strain}\n' for stress, strain in new_stress_strain_data])
    new_lines.extend(material_inp_content[end_line:])

    # Write the updated material.inp file
    with open(filePath, 'w') as file:
        file.writelines(new_lines)

def replace_maxDisp_geometry_inp(filePath, maxTargetDisplacement):
    with open(filePath, 'r') as geometry_inp:
        geometry_inp_content = geometry_inp.readlines()
    start_line = None
    end_line = None
    for i, line in enumerate(geometry_inp_content[-60:]):
        if line.startswith('*Boundary, amplitude'):
            original_index = len(geometry_inp_content) - 60 + i
            start_line = original_index + 1
            end_line = original_index + 2
            break

    if start_line is None or end_line is None:
        raise ValueError('Could not find the *Boundary, amplitude displacement section')

    new_disp_data = f"Disp, 2, 2, {maxTargetDisplacement}\n"

    new_lines = []
    new_lines.extend(geometry_inp_content[:start_line])
    new_lines.extend([new_disp_data])
    new_lines.extend(geometry_inp_content[end_line:])

    with open(filePath, 'w') as file:
        file.writelines(new_lines)

def replace_materialName_geometry_inp(filePath, materialName):
    with open(filePath, 'r') as geometry_inp:
        geometry_inp_content = geometry_inp.readlines()
    start_line = None
    end_line = None
    for i, line in enumerate(geometry_inp_content[-100:]):
        if line.startswith('*INCLUDE, INPUT='):
            original_index = len(geometry_inp_content) - 100 + i
            start_line = original_index
            end_line = original_index + 1
            break

    if start_line is None or end_line is None:
        raise ValueError('Could not find the *INCLUDE, INPUT= section')

    new_material_data = f"*INCLUDE, INPUT={materialName}\n"

    new_lines = []
    new_lines.extend(geometry_inp_content[:start_line])
    new_lines.extend([new_material_data])
    new_lines.extend(geometry_inp_content[end_line:])

    with open(filePath, 'w') as file:
        file.writelines(new_lines)