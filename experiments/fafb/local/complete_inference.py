from __future__ import  print_function
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call

sys.path.append('/groups/saalfeld/home/heinrichl/Projects/simpleference')
from simpleference.inference.util import get_offset_lists
import z5py


def single_inference(gpu, gpu_device):
    call(['./run_inference.sh', str(gpu), str(gpu_device)])
    return True

    # run multiprocessed inference


def complete_inference(gpu_list, gpu_device_list):
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
        tasks = [pp.submit(single_inference, gpu, gpu_device) for gpu_device, gpu in zip(gpu_device_list,gpu_list)]
        result = [t.result() for t in tasks]

        if all(result):
            print("All gpu's finished inference properly.")
        else:
            print("WARNING: at least one process didn't finish properly.")


if __name__ == '__main__':
    gpu_list = [5]#list(range(40,48))#[26,27]#list(range(40, 48))
    gpu_device_list = [2]#list(range(8))#[1,2]
    complete_inference(gpu_list, gpu_device_list)