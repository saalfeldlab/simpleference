#!/bin/sh

# Inputs:
# Sample - id of sample that will be predicted
# GPU - id of the gpu used for inference

PRED_PATH=$HOME/projects/simpleference/experiments/cremi-pytorch

# export all relevant pythonpaths
SIMPLEFERENCE_PATH=$HOME/projects/simpleference
INFERNO_PATH=$HOME/projects/inferno
NFIRE_PATH=$HOME/projects/neurofire
SKUNK_PATH=$HOME/projects/neuro-skunkworks

export PYTHONPATH=${SIMPLEFERENCE_PATH}:${INFERNO_PATH}:${NFIRE_PATH}:${SKUNK_PATH}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=$2
python -u ${PRED_PATH}/run_inference.py $1 $2
