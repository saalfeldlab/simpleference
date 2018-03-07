#!/usr/bin/env bash 
DOCKER_IMAGE="neptunes5thmoon/gunpowder:v0.3-pre6-dask1" 

export CONTAINER_NAME=$(basename $PWD-prediction-35)
export USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Projects/mygunpowder/gunpowder)
SIMPLEFERENCE_PATH=$(readlink -f $HOME/Projects/simpleference)
PRED_PATH=$(readlink -f $HOME/Projects/simpleference/experiments/fafb)
Z_PATH=$(readlink -f $HOME/../papec/Work/my_projects/z5/bld27/python)

nvidia-docker rm -f $NAME

nvidia-docker \
run --rm \
--cgroup-parent=$(cat /proc/self/cpuset) \
--name ${CONTAINER_NAME} \
-u `id -u $USER`:`id -g $USER` \
-v $(pwd):/workspace \
-v /groups/saalfeld:/groups/saalfeld \
-v /nrs/saalfeld/:/nrs/saalfeld \
-w /workspace \
${DOCKER_IMAGE} \
/bin/bash -c "export CUDA_VISIBLE_DEVICES=3; export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=2000; export PYTHONPATH=${GUNPOWDER_PATH}:${SIMPLEFERENCE_PATH}:${Z_PATH}:\$PYTHONPATH; python -u ${PRED_PATH}/run_inference.py 35"

wait
