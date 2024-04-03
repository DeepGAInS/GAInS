#!/bin/bash
#SBATCH -J train
#SBATCH -p xhhgnormal
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gpus-per-task=1

CONFIG_FILE="/work/home/acvwd4uw3y181/rsliu/detectron2/tools/baseline_mask_rcnn_R_50_FPN_1x.yaml"


srun train_net.py --config-file $CONFIG_FILE >> kaggle_maskrcnn.log