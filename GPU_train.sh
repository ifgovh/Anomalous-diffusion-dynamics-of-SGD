#!/bin/bash
#PBS -P THAP
#PBS -N SArrayJobs
#PBS -q defaultQ
#PBS -l select=1:ncpus=1:ngpus=1:mem=24gb
#PBS -l walltime=150:50:59
#PBS -e PBSout/
#PBS -o PBSout/
##PBS -J 1-2
PBS_ARRAY_INDEX=2
cd ~
source tf/bin/activate
cd "$PBS_O_WORKDIR"
params=`sed "${PBS_ARRAY_INDEX}q;d" job_params`
param_array=( $params )

# training DNN
for i in {1..10}
do
	python -m main --model=${param_array[i*5]} --lr=${param_array[i*5+1]} --epochs=${param_array[i*5+2]}  --batch_size=${param_array[i*5+3]} --dataset=${param_array[i*5+4]}
done





