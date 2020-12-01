#!/bin/bash
#PBS -N SArrayJobs
#PBS -q defaultQ
#PBS -l select=1:ncpus=1:ngpus=1:mem=24gb
#PBS -l walltime=100:50:59
#PBS -e PBSout/
#PBS -o PBSout/
#PBS -J 1-8

DATA_DIR="/path/to/this/repo"
cd ${DATA_DIR}

params=`sed "${PBS_ARRAY_INDEX}q;d" job_params`
param_array=( $params )

# training DNN
python -m main --model=${param_array[0]} --epochs=500  --batch_size=1024

# hessian
python hessian.compute_hessian_eig_GZ.py --cuda --batch_size=128 --model=${param_array[0]} --model_folder='path/to/trained_nets/'${param_array[0]}'_sgd_lr=0.1_bs=1024_wd=0_mom=0_save_epoch=1' --num_eigenthings=20

# MATLAB post analysis
MATLAB_SOURCE_PATH="/path/to/post_analysis"
MATLAB_PROCESS_FUNC="SGD_analysis_step_level"

matlab  -nodisplay  -r "cd('${MATLAB_SOURCE_PATH1}'), addpath(genpath(cd)), cd('${PBS_O_WORKDIR}') , \
                                                   cd('${MATLAB_SOURCE_PATH2}'), addpath(genpath(cd)), cd('${PBS_O_WORKDIR}'), \
						   cd('${DATA_DIR}'), ${MATLAB_PROCESS_FUNC}(${PBS_ARRAY_INDEX}), exit"

