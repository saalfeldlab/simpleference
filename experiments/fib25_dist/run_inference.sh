#!/usr/bin/env bash

# Inputs:
# GPU - id of the gpu used for inference
# Iteration - iteration of the network used

export NAME=$(basename $PWD-prediction-$1)
export USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Projects/mygunpowder/gunpowder)
SIMPLEFERENCE_PATH=$(readlink -f $HOME/Projects/git_repos/simpleference)
PRED_PATH=$(readlink -f $HOME/Projects/git_repos/simpleference/experiments/fib25_dist)

docker rm -f $NAME

nvidia-docker run --rm \
    -u ${USER_ID} \
    -v /groups/saalfeld:/groups/saalfeld \
    -v /nrs/saalfeld:/nrs/saalfeld/ \
    -w ${PWD} \
    --name ${NAME} \
    funkey/gunpowder:v0.3-pre5 \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=${1}; PYTHONPATH=${GUNPOWDER_PATH}:${SIMPLEFERENCE_PATH}:\$PYTHONPATH;
    python -u ${PRED_PATH}/run_inference.py ${1} ${2}"
