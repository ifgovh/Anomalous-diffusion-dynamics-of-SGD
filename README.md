This repository contains the PyTorch code for the paper 

<!-- Guozhang Chen, Kevin Qu, Pulin Gong,  -->
Anomalous diffusion dynamics of learning in deep neural networks, 2020

As the SGD can be regarded as a random walker moves on a non-convex and high-dimensional loss-landscape, we use the mean-squared displacement (MSD) to quantify the diffusive process of the SGD training process.
Furthermore, we found that the anomalous diffusion dynamics of SGD is due to the interaction of SGD and fractal-like loss-landscape.

![schematic](doc/images/schematic.png)

Given a network architecture, this tool characterizes the anomalous diffusion of SGD dynamic during the training process of DNNs and illustrate the fractal-like structure of loss-landscape.

# Main pre-requested libraries:
PyTorch > 1.0
numpy 1.15.1
h5py 2.7.0
scipy 0.19
[pytorch-hessian-eigenthings](https://github.com/noahgolmant/pytorch-hessian-eigenthings)

# To reproduce the results in the paper, one just need run below commands or simply submit the pbs script in /PBS_script/all_in_one.sh.

## training DNN
python -m get_gradient_weight.main --model='resnet14' --epochs=500  --batch_size=1024

## hessian
python hessian.compute_hessian_eig_GZ.py --cuda --batch_size=128 --model='resnet14' --model_folder='path/to/trained_nets/resnet14_sgd_lr=0.1_bs=1024_wd=0_mom=0_save_epoch=1' --num_eigenthings=20

## MATLAB post analysis
MATLAB_SOURCE_PATH="/path/to/post_analysis"
MATLAB_PROCESS_FUNC="post_analysis"

matlab  -nodisplay  -r "cd('${MATLAB_SOURCE_PATH1}'), addpath(genpath(cd)), cd('${PBS_O_WORKDIR}') , \
                                                   cd('${MATLAB_SOURCE_PATH2}'), addpath(genpath(cd)), cd('${PBS_O_WORKDIR}'), \
						   cd('${DATA_DIR}'), ${MATLAB_PROCESS_FUNC}(${PBS_ARRAY_INDEX}), exit"


## Code Reference
[1] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein. On large-batch training for deep learning: Generalization gap and sharp minima. ICLR, 2017.


# Citation
If you find this code useful in your research, please cite:

@inproceedings{visualloss,
  title={Anomalous diffusion dynamics of learning in deep neural networks},
  <!-- author={Chen, Guozhang and Qu, Kevin and Gong, Pulin}, -->
  booktitle={under review},
  year={2020}
}