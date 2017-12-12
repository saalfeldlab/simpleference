#!/bin/sh

# Inputs:
# RawPath - path to raw data
# OutPath - path to store the prediction
# NetFolder - directory with network weights
# GPU  - id of the gpu used for inference
# Iteration - iteration of the network used

export NAME=$(basename $PWD-prediction-$4)
export USER_ID=${UID}

# Path to custom gunpowder
GUNPOWDER_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder)

# Path to the simpleference repository
SIMPLEFERENCE_PATH=$(readlink -f $HOME/Work/my_projects/nnets/simpleference)

# Path to this folder
PRED_PATH=$(readlink -f $HOME/Work/my_projects/nnets/simpleference/examples)

# Path to z5 python bindings
Z_PATH=$(readlink -f $HOME/Work/my_projects/z5/bld27/python)

nvidia-docker rm -f $NAME

nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups/saalfeld/home/papec:/groups/saalfeld/home/papec \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.3-pre5 \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=$4; PYTHONPATH=${GUNPOWDER_PATH}:${SIMPLEFERENCE_PATH}:${Z_PATH}:\$PYTHONPATH; python -u ${PRED_PATH}/run_inference.py $1 $2 $3 $4 $5 "
