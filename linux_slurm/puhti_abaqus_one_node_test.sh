#!/bin/bash -l
# Author: Xuan Binh
#SBATCH --job-name=abaqus_OneNodeSmall
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --partition=test
#SBATCH --account=project_2007935
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

unset SLURM_GTIDS
module purge
module load abaqus/2023	

### Change to the work directory
fullpath=$(sed -n 1p linux_slurm/array_file.txt) 
cd ${fullpath}

CPUS_TOTAL=$(( $SLURM_NTASKS*$SLURM_CPUS_PER_TASK ))

abq2023 job=geometry input=geometry.inp parallel=domain domains=$CPUS_TOTAL cpus=$CPUS_TOTAL -verbose 2 interactive

# run postprocess.py after the simulation completes
abq2023 cae noGUI=postprocess.py