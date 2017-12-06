#!/bin/sh

# Inputs:
# GPU - id of the gpu used for inference
# Iteration - iteration of the network used
# GPU-OFFSET

export NAME=$(basename $PWD-prediction-$1)
export USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder)
SIMPLEFERENCE_PATH=$(readlink -f $HOME/Work/my_projects/nnets/simpleference)
PRED_PATH=$(readlink -f $HOME/Work/my_projects/nnets/simpleference/experiments/sampleE-tf)
Z_PATH=$(readlink -f $HOME/Work/my_projects/z5/bld/python)

nvidia-docker rm -f $NAME

nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups/saalfeld/home/papec:/groups/saalfeld/home/papec \
    -v /groups/saalfeld/saalfeldlab:/groups/saalfeld/saalfeldlab \
    -v /nrs/saalfeld/:/nrs/saalfeld \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.3-pre5 \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=$1; PYTHONPATH=${GUNPOWDER_PATH}:${SIMPLEFERENCE_PATH}:${Z_PATH}:\$PYTHONPATH; python -u ${PRED_PATH}/run_inference.py $1 $2 $3"
