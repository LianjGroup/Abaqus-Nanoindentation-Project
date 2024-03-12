import json
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from datetime import datetime
from modules.stoploss import *
from modules.calculation import *
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

from scipy.optimize import minimize



def train_classifiers(nonconverging_combined_objective_value_to_param_FD_Curves, 
                      converging_combined_objective_value_to_param_FD_Curves, 
                      paramConfig,
                      objectives):

    np.random.seed(42) # For reproducibility

    classifiers = {}

    for objective in objectives:    
        converging_FD_Curves = nonconverging_combined_objective_value_to_param_FD_Curves[objective]
        nonconverging_FD_Curves = converging_combined_objective_value_to_param_FD_Curves[objective]
        
        convergingLabels = [0 for i in range (len(converging_FD_Curves))]
        nonconvergingLabels = [1 for i in range (len(nonconverging_FD_Curves))]

        Y = np.array(convergingLabels + nonconvergingLabels)

        #convergingFeatures = [[paramValue for paramName, paramValue in paramTuples] for paramTuples in converging_FD_Curves.keys()]
        #nonconvergingFeatures = [[paramValue for paramName, paramValue in paramTuples] for paramTuples in nonconverging_FD_Curves.keys()]
        
        convergingFeatures = [minmax_scaler(paramTuples, paramConfig) for paramTuples in converging_FD_Curves.keys()]
        nonconvergingFeatures = [minmax_scaler(paramTuples, paramConfig) for paramTuples in nonconverging_FD_Curves.keys()]

        X = np.array(convergingFeatures + nonconvergingFeatures)

        # Creating the SVM classifier
        clf = SVC(C=1.0, kernel='linear', probability=True)  # Using a linear kernel

        # Training the classifier
        clf.fit(X, Y)

        # Making predictions
        y_pred = clf.predict(X)
        
        classifiers[objective] = clf
        # Evaluating the classifier
        accuracy = accuracy_score(Y, y_pred)
        # predict probabilities
        #y_pred_prob = clf.predict_proba(X)
        #print(y_pred_prob)
        print(f"Classifying accuracy for {objective}: {accuracy*100:.2f}%")

    # print("Hello")
    # time.sleep(180)
    return classifiers

def train_linear_models(targetCenters, 
                        converging_combined_objective_value_to_param_FD_Curves, 
                        paramConfig, 
                        objectives):
    linearModels = {}
    for objective in objectives:
        targetCenter = targetCenters[objective]
        converging_FD_Curves = converging_combined_objective_value_to_param_FD_Curves[objective]
        #X = np.array([[paramValue for paramName, paramValue in paramTuples] for paramTuples in converging_FD_Curves.keys()])
        #print(X)
        X = np.array([minmax_scaler(paramTuples, paramConfig) for paramTuples in converging_FD_Curves.keys()])
        # print(X)
        y = []
        for dispForce in converging_FD_Curves.values():
            simCenter = find_sim_center(dispForce)
            loss_value = lossFD_SOO(targetCenter, simCenter)
            # if loss_value < 0:
            #     print("Loss value is negative")
            #     print(targetCenter)
            #     print(simCenter)
            #     print(loss_value)
            #     print("=====================================================")
            y.append(loss_value)
        y = np.array(y)

        # Take log transformation, as prediction in the future could be negative
        # When new prediction is negative, we can exponentiate it to get the positive value

        # y_log = np.log(y + 0.00001) # Adding a small value to avoid log(0)
        
        #model = LinearRegression(fit_intercept=True).fit(X, y_log)
        model = LinearRegression(fit_intercept=True).fit(X, y)
        #y_pred_log = model.predict(X)
        #y_pred = np.exp(y_pred_log)
        y_pred = model.predict(X)
        MSE = np.mean((y - y_pred) ** 2)
        print(f"RMSE for {objective}: {np.sqrt(MSE):.2f}")
        linearModels[objective] = model
    return linearModels

def minimize_custom_loss_function(classifiers, regressionModels, paramConfig, objectives):
    bounds = np.array([(0, 1) for i in range(len(paramConfig))])
    x0 = np.array([0.5 for i in range(len(paramConfig))])
    res = minimize(custom_lossFD, x0, args=(classifiers, regressionModels, objectives), 
                   bounds=bounds, method='Powell', # Nelder-Mead
                   options={'disp': False, 'maxiter': 1000000})
    scaled_X = res.x
    optimal_X = [de_minmax_scaler(scaled_X, paramConfig)]

    next_paramDict = {}
    # print(optimal_X)
    for i, paramName in enumerate(paramConfig):
        next_paramDict[paramName] = optimal_X[0][i]

    return next_paramDict

def custom_lossFD(X, classifiers, regressionModels, objectives):
    objectiveLosses = []
    for objective in objectives:
        classifierObjective = classifiers[objective]
        regressionModelObjective = regressionModels[objective]
        converging_prob, nonconverging_prob = classifierObjective.predict_proba(X.reshape(1, -1))[0]
        euclideanLoss = regressionModelObjective.predict(X.reshape(1, -1))        

        if nonconverging_prob - converging_prob > 0.2:
            objectiveLosses.append(1e12)
        else:
            if euclideanLoss[0] < 0:
                objectiveLosses.append(1e12)
            else: 
                objectiveLosses.append(euclideanLoss[0])
                
    return np.sum(objectiveLosses)

# def BayesianLinearRegression(targetCenter, converging_FD_Curves):
#     X = np.array([[paramValue for paramName, paramValue in paramTuples] for paramTuples in converging_FD_Curves.keys()])
#     # print(X)
#     # Standardize X
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
    
#     y = []
#     for dispForce in converging_FD_Curves.values():
#         disp = dispForce["displacement"]
#         force = dispForce["force"]
#         maxForce = np.max(force)
#         maxDispCorresponding = disp[np.argmax(force)]
#         simCenter = {"X": maxDispCorresponding, "Y": maxForce}
#         loss_value = lossFD_SOO(targetCenter, simCenter)
#         y.append(loss_value)
#     y = np.array(y)

#     # Fit the model
#     # Assume y = Xw + epsilon, where w ~ N(m, S) and epsilon ~ N(0, sigma^2 I)
#     # We can use frequentist linear regression to obtain the unknown values above

#     # Fit the model
#     model = LinearRegression(fit_intercept=False).fit(X, y)
#     y_pred = model.predict(X)
#     # The residuals 
#     residuals = y - y_pred
#     # The std of the residuals
#     sigma_epsilon_prior = np.std(residuals)
#     # Now, the mean of the weights is
#     mu_w_prior = model.coef_
#     sigma_w_prior = np.eye(X.shape[1]) * 0.1
    
#     # P(w | y) = N(w | mu_post, SIGMA_post)
    
#     sigma_w_post = np.linalg.inv(1/(sigma_epsilon_prior ** 2) * X.T @ X + np.linalg.inv(sigma_w_prior))
#     mu_w_post = sigma_w_post @ (1/(sigma_epsilon_prior ** 2) * X.T @ y + np.linalg.inv(sigma_w_prior) @ mu_w_prior)
    
#     print("Frequentist linear regression weights: ", model.coef_)
#     print("Bayesian linear regression weights: ", mu_w_post)
#     return mu_w_post
