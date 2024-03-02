#!/bin/bash

#SBATCH --export=ALL
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -e error.e
#SBATCH --mem=10G
#SBATCH -p gpu3
#SBATCH -w wmc-slave-g12

conda activate BCFL
cd /public/zhouchao/BC-enhanced-FL-to-agianst-poison-attack
source /public/zhouchao/.bashrc
conda activate BCFL

srun /public/zhouchao/miniconda3/envs/BCFL/bin/python main.py --datasets mnist_small --model MNISTCNN --aggregate_type fedavg --wandb_log True --lr 1e-2  --epoch_num 200 --wandb_resume True


srun /public/zhouchao/miniconda3/envs/BCFL/bin/python main.py --datasets mnist_small --model MNISTCNN --aggregate_type fedavg --wandb_log True --lr 8e-3  --epoch_num 200


srun /public/zhouchao/miniconda3/envs/BCFL/bin/python main.py --datasets mnist_small --model MNISTCNN --aggregate_type fedavg --wandb_log True --lr 5e-3  --epoch_num 200

