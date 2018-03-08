#!/usr/bin/env bash 
set -e 

WORK_DIR=$(pwd) 
DOCKER_IMAGE="neptunes5thmoon/gunpowder:v0.3-pre6-dask1" 

USER_ID=${UID}
CONTAINER_NAME=$(basename ${USER}-${WORK_DIR}-prediction-30-${RANDOM})

GUNPOWDER_PATH=$(readlink -f $HOME/Projects/mygunpowder/gunpowder)
SIMPLEFERENCE_PATH=$(readlink -f $HOME/Projects/simpleference)
PRED_PATH=$(readlink -f $HOME/Projects/simpleference/experiments/fafb)
Z_PATH=$(readlink -f $HOME/../papec/Work/my_projects/z5/bld27/python)

teardown() {
  trap - SIGINT SIGTERM 
  echo "run docker: Stopping container ${CONTAINER_NAME}, killing after 5s..."
  docker stop -t5 ${CONTAINER_NAME}
  echo "run docker: Container ${CONTAINER_NAME} stopped."
}

trap teardown SIGINT SIGTERM

export NV_GPU=$CUDA_VISIBLE_DEVICES
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
/bin/bash -c "sleep 5; export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=2000; export PYTHONPATH=${GUNPOWDER_PATH}:${SIMPLEFERENCE_PATH}:${Z_PATH}:\$PYTHONPATH; python -u ${PRED_PATH}/run_inference.py 30"

wait
