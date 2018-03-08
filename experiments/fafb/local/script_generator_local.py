import os
import stat


def write_scripts(gpu_list, gpu_device_list):
    for gpu, gpu_device in zip(gpu_list, gpu_device_list):
        bash_script = \
    '#!/usr/bin/env bash \n\
DOCKER_IMAGE="neptunes5thmoon/gunpowder:v0.3-pre6-dask1" \n\
\n\
export CONTAINER_NAME=$(basename $PWD-prediction-%d)\n\
export USER_ID=${UID}\n\
GUNPOWDER_PATH=$(readlink -f $HOME/Projects/mygunpowder/gunpowder)\n\
SIMPLEFERENCE_PATH=$(readlink -f $HOME/Projects/simpleference)\n\
PRED_PATH=$(readlink -f $HOME/Projects/simpleference/experiments/fafb)\n\
Z_PATH=$(readlink -f $HOME/../papec/Work/my_projects/z5/bld27/python)\n\
\n\
nvidia-docker rm -f $NAME\n\
\n\
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
/bin/bash -c "export CUDA_VISIBLE_DEVICES=%d; export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=2000; export PYTHONPATH=${' \
    'GUNPOWDER_PATH}:${SIMPLEFERENCE_PATH}:${Z_PATH}:\$PYTHONPATH; python -u ${' \
  'PRED_PATH}/run_inference.py %d"\n\
\nwait\n'% (gpu, gpu_device, gpu)

        with open('local/scripts/run_inference_%d.sh'%gpu, 'w') as f:
            print(f.name)
            dirc = os.path.join(os.getcwd(), f.name)
            print(dirc)
            st = os.stat(dirc)
            os.chmod(dirc, st.st_mode | stat.S_IEXEC)
            os.chmod(dirc, st.st_mode | stat.S_IWGRP)
            os.chmod(dirc, st.st_mode | stat.S_IRGRP)
            f.write(bash_script)


def write_scripts_choose_gpu(gpu_list):
    for gpu in gpu_list:
        bash_script = \
    '#!/usr/bin/env bash \n\
DOCKER_IMAGE="neptunes5thmoon/gunpowder:v0.3-pre6-dask1" \n\
\n\
export CONTAINER_NAME=$(basename $PWD-prediction-%d)\n\
export USER_ID=${UID}\n\
GUNPOWDER_PATH=$(readlink -f $HOME/Projects/mygunpowder/gunpowder)\n\
SIMPLEFERENCE_PATH=$(readlink -f $HOME/Projects/simpleference)\n\
PRED_PATH=$(readlink -f $HOME/Projects/simpleference/experiments/fafb)\n\
Z_PATH=$(readlink -f $HOME/../papec/Work/my_projects/z5/bld27/python)\n\
\n\
nvidia-docker rm -f $NAME\n\
\n\
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
/bin/bash -c "export CUDA_VISIBLE_DEVICES=$1; export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=2000; export PYTHONPATH=${' \
    'GUNPOWDER_PATH}:${SIMPLEFERENCE_PATH}:${Z_PATH}:\$PYTHONPATH; python -u ${' \
  'PRED_PATH}/run_inference.py %d"\n\
\nwait\n'% (gpu, gpu)

        with open('local/scripts_choose_gpu_device/run_inference_%d.sh'%gpu, 'w') as f:
            print(f.name)
            dirc = os.path.join(os.getcwd(), f.name)
            print(dirc)
            st = os.stat(dirc)
            os.chmod(dirc, st.st_mode | stat.S_IEXEC)
            os.chmod(dirc, st.st_mode | stat.S_IWGRP)
            os.chmod(dirc, st.st_mode | stat.S_IRGRP)
            f.write(bash_script)
