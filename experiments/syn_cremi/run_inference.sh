#!/bin/sh

# Inputs:
# Path - Path to the N5 dataset with raw data and mask
# GPU - id of the gpu used for inference
# Iteration - iteration of the network used

export NAME=$(basename $PWD-prediction-$2)
export USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Projects/mygunpowder/gunpowder)
SIMPLEFERENCE_PATH=$(readlink -f $HOME/Projects/simpleference)
PRED_PATH=$(readlink -f $HOME/Projects/simpleference/experiments/syn_cremi)
Z_PATH=$(readlink -f $HOME/../papec/Work/my_projects/z5/bld27/python)

nvidia-docker rm -f $NAME

nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups/saalfeld:/groups/saalfeld \
    -v /nrs/saalfeld/:/nrs/saalfeld \
    -w /workspace \
    --name $NAME \
    neptunes5thmoon/gunpowder:v0.3-pre6-dask1 \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=$2;
    export PYTHONPATH=${GUNPOWDER_PATH}:${SIMPLEFERENCE_PATH}:${Z_PATH}:\$PYTHONPATH;
    python -u ${PRED_PATH}/run_inference.py $1 $2 $3"