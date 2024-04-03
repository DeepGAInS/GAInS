#!/bin/bash
#SBATCH -J demo
#SBATCH -p xhacnormalb
#SBATCH -N 1
#SBATCH -n 64


srun demo_dataset.py