#!/bin/bash -l
# Author: Youngbin Pyo
#SBATCH --job-name=3198_0_1
#SBATCH --error=abaqus.err
#SBATCH --output=abaqus.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --partition=small
#SBATCH --account=project_2004956
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youngbin.pyo@aalto.fi

unset SLURM_GTIDS

# Old Intel compilers
module load intel-oneapi-compilers-classic
# module load intel-oneapi-compilers
# module load gcc

# Loading intel MPI
module load openmpi

module load abaqus
module load python-data

cd $PWD

CPUS_TOTAL=$(( $SLURM_NTASKS*$SLURM_CPUS_PER_TASK ))

# Default path of abaqus_v6.env
# /appl/soft/eng/simulia/EstProducts/2023/linux_a64/SMA/site/lnx86_64.env

abaqus job=nano_umat input=cubecorner.inp user=umat_BCC_CRYSTAL_Rate_Dependent__NewHardL_Local cpus=$CPUS_TOTAL double=both output_precision=full -verbose 2 interactive

abaqus cae noGUI=postprocess.py