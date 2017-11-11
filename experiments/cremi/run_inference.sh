#!/bin/sh

# Inputs:
# GPU - id of the gpu used for inference
# Sample - id of sample that will be predicted
# Iteration - iteration of the network used

export NAME=$(basename $PWD-prediction-$2)
export USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder)
SIMPLEFERENCE_PATH=$(readlink -f $HOME/Work/my_projects/nnets/simpleference)
PRED_PATH=$(readlink -f $HOME/Work/my_projects/nnets/simpleference/experiments/cremi)

nvidia-docker rm -f $NAME

nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups/saalfeld/home/papec:/groups/saalfeld/home/papec \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.3-pre2 \
    /bin/bash -c "PYTHONPATH=${GUNPOWDER_PATH}:${SIMPLEFERENCE_PATH}:\$PYTHONPATH; python -u ${PRED_PATH}/run_inference.py $1 $2 $3"
