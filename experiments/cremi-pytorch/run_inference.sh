#!/bin/sh

# Inputs:
# Sample - id of sample that will be predicted
# GPU - id of the gpu used for inference

PRED_PATH=$(readlink -f $HOME/Work/my_projects/nnets/simpleference/experiments/cremi-pytorch)

# export all relevant pythonpaths
SIMPLEFERENCE_PATH=$(readlink -f $HOME/Work/my_projects/nnets/simpleference)
Z_PATH=$(readlink -f $HOME/Work/my_projects/z5/bld/python)
INFERNO_PATH=$(readlink -f $HOME/Work/my_projects/nnets/inferno)
NFIRE_PATH=$(readlink -f $HOME/Work/my_projects/nnets/neurofire)
SKUNK_PATH=$(readlink -f $HOME/Work/my_projects/nnets/neuro-skunkworks)

PYTHONPATH=${SIMPLEFERENCE_PATH}:${Z_PATH}:${INFERNO_PATH}:${NFIRE_PATH}:${SKUNK_PATH}:\$PYTHONPATH

export CUDA_VISIBLE_DEVICES=$2
python -u ${PRED_PATH}/run_inference.py $1 $2
