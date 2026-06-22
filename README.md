# Abaqus Nanoindentation Project

This repository contains an automated Abaqus-based workflow for calibrating crystal plasticity parameters using nanoindentation force-displacement curves.

## Overview

The project is designed for finite element nanoindentation simulations in Abaqus, with a focus on crystal plasticity parameter calibration. The workflow compares simulated force-displacement curves against target experimental curves and iteratively searches for improved material parameters.

The pipeline includes:

- reading global configuration files,
- preparing experimental target force-displacement curves,
- generating initial parameter sets,
- running Abaqus simulations,
- extracting simulated force-displacement curves,
- filtering converged and non-converged simulations,
- iteratively updating parameters until the target deviation criterion is satisfied.

## Repository Structure

```text
.
├── configs/          # Global configuration files
├── linux_slurm/      # SLURM scripts for running Abaqus jobs on HPC systems
├── modules/          # Python helper modules for simulation, I/O, and curve processing
├── optimizers/       # Optimization and parameter-search routines
├── paramInfo/        # Parameter information and bounds
├── results/          # Simulation and calibration results
├── simulations/      # Generated simulation folders
├── targets/          # Experimental target force-displacement curves
├── templates/        # Abaqus input templates
├── notebooks/        # Analysis notebooks
├── pipeline.py       # Main workflow driver
└── requirements.txt  # Python dependencies
```

## Main Workflow

The complete pipeline can be started from:

```bash
python pipeline.py
```

The workflow follows these stages:

1. `stage0_configs.py`  
   Reads configuration files and initializes the project directory structure.

2. `stage1_prepare_targetCurve.py`  
   Reads experimental nanoindentation force-displacement curves from the target folder.

3. `stage2_run_initialSims.py`  
   Generates initial parameter sets and submits Abaqus simulations.

4. `stage3_prepare_simCurves.py`  
   Reads and processes simulated force-displacement curves.

5. `stage4_iterative_calibration.py`  
   Performs iterative parameter calibration until the deviation criterion is satisfied.

## Requirements

Python dependencies are listed in `requirements.txt`.

Main dependencies include:

```text
numpy
pandas
matplotlib
scikit-learn
scipy
torch
botorch
openpyxl
prettytable
sobol-seq
bayesian-optimization
```

The workflow also requires:

- Abaqus,
- a compatible Fortran compiler for user-material simulations,
- SLURM-based job submission if running on an HPC cluster.

## Notes

This repository is intended for research use in computational mechanics, finite element simulation, and crystal plasticity calibration. The included scripts assume a specific Abaqus nanoindentation template structure and may require adaptation before use on a different machine, cluster, or material system.

## Maintainers

Developed within the Lian group for research on nanoindentation, finite element simulation, and crystal plasticity modeling.
