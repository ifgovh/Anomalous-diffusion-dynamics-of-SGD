#!/bin/bash
#PBS -N SArrayJobs
#PBS -q defaultQ
#PBS -l select=1:ncpus=1:ngpus=1:mem=24gb
#PBS -l walltime=100:50:59
#PBS -e PBSout/
#PBS -o PBSout/
#PBS -J 1-8

DATA_DIR="/project/THAP/Anomalous-diffusion-dynamics-of-SGD"
cd ${DATA_DIR}

params=`sed "${PBS_ARRAY_INDEX}q;d" job_params`
param_array=( $params )

# training DNN
for i in {0..7}
do
	python -m main --model=${param_array[i*5+1]} --lr=${param_array[i*5+2]} --epochs=${param_array[i*5+3]}  --batch_size=${param_array[i*5+4]} --dataset=${param_array[i*5+5]}
done





