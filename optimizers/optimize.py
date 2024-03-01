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

def LinearRegression(FD_Curves, targetCurve, yieldingIndex, paramConfig,iteration):
    pass