#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -e error_scale.e
#SBATCH --mem=10G
#SBATCH -p gpu3
#SBATCH -w wmc-slave-g11
conda activate BCFL
cd /public/zhouchao/BC-enhanced-FL-to-agianst-poison-attack
srun /public/zhouchao/miniconda3/envs/BCFL/bin/python main.py --datasets mnist_small --model MNISTCNN --aggregate_type fedavg --eva_type ACC --grad_scale_num 3 --wandb_log True 

srun /public/zhouchao/miniconda3/envs/BCFL/bin/python main.py --datasets mnist_small --model MNISTCNN --aggregate_type fedavg --validation_nodes_num 2 --eva_type ACC --grad_scale_num 3 --wandb_log True 

srun /public/zhouchao/miniconda3/envs/BCFL/bin/python main.py --datasets mnist_small --model MNISTCNN --aggregate_type krum --eva_type ACC --grad_scale_num 3 --wandb_log True 

srun /public/zhouchao/miniconda3/envs/BCFL/bin/python main.py --datasets mnist_small --model MNISTCNN --aggregate_type median --eva_type ACC --grad_scale_num 3 --wandb_log True 

srun /public/zhouchao/miniconda3/envs/BCFL/bin/python main.py --datasets mnist_small --model MNISTCNN --aggregate_type trimmed_mean --eva_type ACC --grad_scale_num 3 --wandb_log True 

