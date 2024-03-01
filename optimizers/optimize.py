import json
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from datetime import datetime
from modules.stoploss import *
import time

import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils import standardize
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def SOO_write_BO_json_log(FD_Curves, targetCurve, yieldingIndex, paramConfig,iteration):
    # Write the BO log file
    # Each line of BO logging json file looks like this
    # {"target": <loss value>, "params": {"params1": <value1>, ..., "paramsN": <valueN>}, "datetime": {"datetime": "2023-06-02 18:26:46", "elapsed": 0.0, "delta": 0.0}}
    # FD_Curves is a dictionaries
    # where each element is of form (parameterTuples) => {"displacement": <np.array>, "force": <np.array>}
    # targetCurve is a dictionary of form {"displacement": <np.array>, "force": <np.array>}

    # Construct the json file line by line for each element in FD_Curves
    # Each line is a dictionary
    
    # Delete the json file if it exists
    if os.path.exists(f"optimizers/logs.json"):
        os.remove(f"optimizers/logs.json")

    for paramsTuple, dispforce in FD_Curves.items():
        # Construct the dictionary
        line = {}
        # Note: BO in Bayes-Opt tries to maximize, so you should use the negative of the loss function.
        line["target"] = -lossFD(targetCurve["displacement"][yieldingIndex:], targetCurve["force"][yieldingIndex:], dispforce["force"][yieldingIndex:],iteration)
        line["params"] = dict(paramsTuple)
        for param in paramConfig:
            line["params"][param] = line["params"][param]/paramConfig[param]["exponent"] 
        line["datetime"] = {}
        line["datetime"]["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line["datetime"]["elapsed"] = 0.0
        line["datetime"]["delta"] = 0.0

        # json file has not exist yet
        # Write the dictionary to json file
        with open(f"optimizers/logs.json", "a") as file:
            json.dump(line, file)
            file.write("\n")

def MOO_write_BO_json_log(combined_interpolated_params_to_geoms_FD_Curves_smooth, targetCurves, geometries, geometryWeights, yieldingIndices, paramConfig,iteration):
    
    # Delete the json file if it exists
    if os.path.exists(f"optimizers/logs.json"):
        os.remove(f"optimizers/logs.json")

    for paramsTuple, geometriesToForceDisplacement in combined_interpolated_params_to_geoms_FD_Curves_smooth.items():
        # Construct the dictionary
        line = {}
        # Note: BO in Bayes-Opt tries to maximize, so you should use the negative of the loss function.
        loss = 0
        for geometry in geometries:
            yieldingIndex = yieldingIndices[geometry]
            loss += - geometryWeights[geometry] * lossFD(targetCurves[geometry]["displacement"][yieldingIndex:], targetCurves[geometry]["force"][yieldingIndex:], geometriesToForceDisplacement[geometry]["force"][yieldingIndex:],iteration)
        line["target"] = loss
        line["params"] = dict(paramsTuple)
        for param in paramConfig:
            line["params"][param] = line["params"][param]/paramConfig[param]["exponent"] 
        line["datetime"] = {}
        line["datetime"]["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line["datetime"]["elapsed"] = 0.0
        line["datetime"]["delta"] = 0.0

        # json file has not exist yet
        # Write the dictionary to json file
        with open(f"optimizers/logs.json", "a") as file:
            json.dump(line, file)
            file.write("\n")

def MOO_suggest_BOTORCH(combined_interpolated_params_to_geoms_FD_Curves_smooth, targetCurves, geometries, yieldingIndices, paramConfig,iteration):
    # Calculate losses and prepare data for model
    params = []
    losses = []
    for param_tuple, geom_to_simCurves in combined_interpolated_params_to_geoms_FD_Curves_smooth.items():
        #print(param_tuple)
        params.append([value for param, value in param_tuple])
        # The minus sign is because BOTORCH tries to maximize objectives, but we want to minimize the loss
        loss_iter = []
        for geometry in geometries:
            yieldingIndex = yieldingIndices[geometry]
            loss_iter.append(- lossFD(
                targetCurves[geometry]["displacement"][yieldingIndex:], 
                targetCurves[geometry]["force"][yieldingIndex:], 
                geom_to_simCurves[geometry]["force"][yieldingIndex:],
                iteration
            ))
        losses.append(loss_iter)

    # Convert your data to the tensor(float 64)
    X = torch.tensor(params, dtype=torch.float64)
    Y = torch.stack([torch.tensor(loss, dtype=torch.float64) for loss in losses])

    # Define the bounds of the search space
    lower_bounds = torch.tensor([paramConfig[param]['lowerBound'] * paramConfig[param]['exponent'] for param in paramConfig.keys()]).float()
    upper_bounds = torch.tensor([paramConfig[param]['upperBound'] * paramConfig[param]['exponent'] for param in paramConfig.keys()]).float()

    bounds = np.vstack([lower_bounds, upper_bounds])

    # Create the MinMaxScaler and fit it to the bounds
    scaler = MinMaxScaler().fit(bounds)

    # Transform the parameters using the fitted scaler
    X_normalized = torch.tensor(scaler.transform(X.numpy()), dtype=torch.float64)

    # Standardize Y to have zero mean and unit variance
    Y_standardized = standardize(Y)

    # Normalise the bounds in accordance to the normalised params
    bounds_normalized = torch.tensor([[0.0]*X_normalized.shape[1], [1.0]*X_normalized.shape[1]])

    # Initialize model
    model = SingleTaskGP(X_normalized, Y_standardized)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Define the acquisition function
    # **Reference Point**

    # qEHVI requires specifying a reference point, which is the lower bound on the objectives used for computing hypervolume. 
    # In this tutorial, we assume the reference point is known. In practice the reference point can be set 
    # 1) using domain knowledge to be slightly worse than the lower bound of objective values, 
    # where the lower bound is the minimum acceptable value of interest for each objective, or 
    # 2) using a dynamic reference point selection strategy.
    ref_point = Y_standardized.max(dim=0).values - 0.01

    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y_standardized)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        
        objective=IdentityMCMultiOutputObjective(),
    )

    # Optimize the acquisition function
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds_normalized,
        q=1,#q: This is the number of points to sample in each step
        num_restarts=10,#num_restarts: This is the number of starting points for the optimization.
        raw_samples=1000,#raw_samples: This is the number of samples to draw when initializing the optimization
    )

    # Unnormalize the candidates
    candidates = torch.tensor(scaler.inverse_transform(candidates.detach().numpy()), dtype=torch.float64)

    #converting to dictionary
    pareto_front = [{param: value.item() for param, value in zip(paramConfig.keys(), next_param)} for next_param in candidates]
    return pareto_front


def MOO_calculate_geometries_weight(targetCurves, geometries):
    geometryWeights = {}
    
    for geometry in geometries:
        targetDisplacement = targetCurves[geometry]["displacement"]
        targetForce = targetCurves[geometry]["force"]
        
        x_start = min(targetDisplacement)
        x_end = max(targetDisplacement)

        # Interpolate the force-displacement curve
        target_FD_func = interp1d(targetDisplacement, targetForce, fill_value="extrapolate")

        # Evaluate the two curves at various points within the x-range boundary
        x_values = np.linspace(x_start, x_end, num=10000)

        y_values = target_FD_func(x_values)

        area = simpson(y_values, x_values)
        geometryWeights[geometry] = 1/np.array(area)
    
    # normalize the weights
    sumWeights = np.sum(list(geometryWeights.values()))
    for geometry in geometryWeights:
        geometryWeights[geometry] = geometryWeights[geometry]/sumWeights

    return geometryWeights