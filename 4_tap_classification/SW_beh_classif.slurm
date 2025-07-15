#!/bin/bash

#SBATCH --job-name=SW_tap_classif
#SBATCH --output=SW_tap_classif.out
#SBATCH --error=SW_tap_classif.err
#SBATCH --mail-user="s3821013@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu-long"
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l4:1

module load ALICE/default
module load shared
module load TensorFlow
module load scikit-learn
module load matplotlib
module load Seaborn

cd /home/s3821013/pipelineTC

python tap_classification_pipeline.py