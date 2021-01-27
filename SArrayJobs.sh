#!/bin/bash
#PBS -P cortical
#PBS -N SArrayJobs
#PBS -q defaultQ
#PBS -l nodes=1:ppn=24
#PBS -l walltime=53:50:59
#PBS -l mem=123GB
#PBS -e PBSout/
#PBS -o PBSout/
##PBS -V
##PBS -J 5-6

PBS_ARRAY_INDEX=7
MATLAB_SOURCE_PATH1="/project/THAP/Anomalous-diffusion-dynamics-of-SGD/post_analysis"
MATLAB_PROCESS_FUNC="SGD_analysis_step_level"
DATA_DIR="trained_nets"
echo ${PBS_ARRAY_INDEX}
set -x
cd "$PBS_O_WORKDIR"

#for i in {1..10}
#do
#	matlab  -singleCompThread -nodisplay  -r "cd('${MATLAB_SOURCE_PATH1}'), addpath(genpath(cd)), cd('${PBS_O_WORKDIR}') , \
#                                                   cd('${MATLAB_SOURCE_PATH2}'), addpath(genpath(cd)), cd('${PBS_O_WORKDIR}'), \
#                                                   cd('${DATA_DIR}'), addpath(genpath(cd)), ${MATLAB_PROCESS_FUNC}(i,${PBS_ARRAY_INDEX}), exit" # Create ${INPUT_APPENDIX} file
#done
matlab  -nodisplay  -r "cd('${MATLAB_SOURCE_PATH1}'), addpath(genpath(cd)), cd('${PBS_O_WORKDIR}') , \
                                                   cd('${DATA_DIR}'), addpath(genpath(cd)), ${MATLAB_PROCESS_FUNC}(${PBS_ARRAY_INDEX}),exit"
