import os
import stat


def write_scripts(gpu_list):
    for gpu in gpu_list:
        bash_script = \
    '#!/usr/bin/env bash \n\
set -e \n\
\n\
WORK_DIR=$(pwd) \n\
DOCKER_IMAGE="neptunes5thmoon/gunpowder:v0.3-pre6-dask1" \n\
\n\
USER_ID=${UID}\n\
CONTAINER_NAME=$(basename ${USER}-${WORK_DIR}-prediction-%d-${RANDOM})\n\
\n\
GUNPOWDER_PATH=$(readlink -f $HOME/Projects/mygunpowder/gunpowder)\n\
SIMPLEFERENCE_PATH=$(readlink -f $HOME/Projects/simpleference)\n\
PRED_PATH=$(readlink -f $HOME/Projects/simpleference/experiments/cremi_validation)\n\
Z_PATH=$(readlink -f $HOME/../papec/Work/my_projects/z5/bld27/python)\n\
\n\
teardown() {\n\
  trap - SIGINT SIGTERM \n\
  echo "run docker: Stopping container ${CONTAINER_NAME}, killing after 5s..."\n\
  docker stop -t5 ${CONTAINER_NAME}\n\
  echo "run docker: Container ${CONTAINER_NAME} stopped."\n\
}\n\
\n\
trap teardown SIGINT SIGTERM\n\
\n\
export NV_GPU=$CUDA_VISIBLE_DEVICES\n\
nvidia-docker \\\n\
run --rm \\\n\
--cgroup-parent=$(cat /proc/self/cpuset) \\\n\
--name ${CONTAINER_NAME} \\\n\
-u `id -u $USER`:`id -g $USER` \\\n\
-v $(pwd):/workspace \\\n\
-v /groups/saalfeld:/groups/saalfeld \\\n\
-v /nrs/saalfeld/:/nrs/saalfeld \\\n\
-w /workspace \\\n\
${DOCKER_IMAGE} \\\n\
/bin/bash -c "sleep 5; export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=2000; export PYTHONPATH=${GUNPOWDER_PATH}:${' \
    'SIMPLEFERENCE_PATH}:${Z_PATH}:\$PYTHONPATH; python -u ${' \
  'PRED_PATH}/run_inference.py %d"\n\
\nwait\n'% (gpu, gpu)

        with open('lsf/scripts/run_inference_%d.sh'%gpu, 'w') as f:
            print(f.name)
            dirc = os.path.join(os.getcwd(), f.name)
            print(dirc)
            st = os.stat(dirc)
            os.chmod(dirc, st.st_mode | stat.S_IEXEC)
            os.chmod(dirc, st.st_mode | stat.S_IWGRP)
            os.chmod(dirc, st.st_mode | stat.S_IRGRP)
            f.write(bash_script)
