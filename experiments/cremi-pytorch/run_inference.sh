#!/bin/sh

# Inputs:
# Sample - id of sample that will be predicted
# GPU - id of the gpu used for inference

PRED_PATH=$HOME/projects/simpleference/experiments/cremi-pytorch

export CUDA_VISIBLE_DEVICES=$2
python -u ${PRED_PATH}/run_inference.py $1 $2
