#!/bin/bash

#SBATCH --job-name=SW_classification
#SBATCH --output=SW_classification.out
#SBATCH --error=SW_classification.err
#SBATCH --mail-user="s3821013@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu-short"
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1

module load ALICE/default
module load TensorFlow
module load scikit-learn

cd /home/s3821013/cnn_cv_pipeline
python cnn_cv_pipeline.py