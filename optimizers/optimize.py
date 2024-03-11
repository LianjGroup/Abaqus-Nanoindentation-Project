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


import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_nonconverging_classifiers(combined_objective_value_to_param_FD_Curves, objectives):
    
    nonconverging_objectives_to_params = {}
    
    for objective in objectives:    
        nonconverging_objectives_to_params[objective] = {}
        nonconverging_objectives_to_params[objective]["paramFeatures"] = []
        nonconverging_objectives_to_params[objective]["nonconvergingLabels"] = []
        
        for params, dispForce in combined_objective_value_to_param_FD_Curves[objective].items():
            paramsList = [paramValue for paramName, paramValue in params]
            nonconverging_objectives_to_params[objective]["paramFeatures"].append(paramsList)
            
            force = dispForce["force"]
            if len(force) < 1000:
                nonconverging_objectives_to_params[objective]["nonconvergingLabels"].append(1)
            else: 
                nonconverging_objectives_to_params[objective]["nonconvergingLabels"].append(0)

    # Convert to numpy 

    for objective in objectives:
        nonconverging_objectives_to_params[objective]["paramFeatures"] = np.array(nonconverging_objectives_to_params[objective]["paramFeatures"])
        nonconverging_objectives_to_params[objective]["nonconvergingLabels"] = np.array(nonconverging_objectives_to_params[objective]["nonconvergingLabels"])        
        # print(nonconverging_objectives_to_params[objective]["paramFeatures"].shape)
        # print(nonconverging_objectives_to_params[objective]["nonconvergingLabels"].shape)

    np.random.seed(42) # For reproducibility

    classifiers = {}

    for objective in objectives:
        X = nonconverging_objectives_to_params[objective]["paramFeatures"]  # Features
        y = nonconverging_objectives_to_params[objective]["nonconvergingLabels"]  # Labels

        # Creating the SVM classifier
        clf = SVC(C=1.0, kernel='linear')  # Using a linear kernel

        # Training the classifier
        clf.fit(X, y)

        # Making predictions
        y_pred = clf.predict(X)
        
        classifiers[objective] = clf
        # Evaluating the classifier
        accuracy = accuracy_score(y, y_pred)

        print(f"Accuracy for {objective}: {accuracy*100:.2f}%")

    # print("Hello")
    # time.sleep(180)
    return classifiers