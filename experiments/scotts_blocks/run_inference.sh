#!/bin/sh

# Inputs:
# Sample - id of sample that will be predicted
# GPU - id of the gpu used for inference
# Iteration - iteration of the network used

export NAME=$(basename $PWD-prediction-$2)
export USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder)
SIMPLEFERENCE_PATH=$(readlink -f $HOME/Work/my_projects/nnets/simpleference)
PRED_PATH=$(readlink -f $HOME/Work/my_projects/nnets/simpleference/experiments/scotts_blocks)
Z_PATH=$(readlink -f $HOME/Work/my_projects/z5/bld27/python)

nvidia-docker rm -f $NAME

nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups/saalfeld/home/papec:/groups/saalfeld/home/papec \
    -v /nrs/saalfeld/:/nrs/saalfeld \
    -w /workspace \
    --name $NAME \
    neptunes5thmoon/gunpowder:v0.3-debug \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=$2; PYTHONPATH=${GUNPOWDER_PATH}:${SIMPLEFERENCE_PATH}:${Z_PATH}:\$PYTHONPATH; python -u ${PRED_PATH}/run_inference.py $1 $2 $3"
