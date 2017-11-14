from __future__ import print_function
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call
from simpleference.inference.util import get_offset_lists
from stitch import stitch


def single_inference(gpu, iteration):
    call(['./run_inference.sh', str(gpu), str(iteration)])
    return True


def complete_inference(gpu_list, iteration):

    # path to the raw data

    raw_path = '/groups/saalfeld/saalfeldlab/larissa/data/fib25/grayscale_sub.h5'
    # make the offset files, that assign blocks to gpus
    save_folder = '/nrs/saalfeld/heinrichl/segmentation/distance_thirdtest/offsets'
    get_offset_lists(raw_path, gpu_list, save_folder, output_shape=(44,)*3, randomize=True)

    # run multiprocessed inference
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
        tasks = [pp.submit(single_inference, gpu, iteration) for gpu in gpu_list]
        result = [t.result() for t in tasks]

    if all(result):
        print("All gpu's finished inference properly.")
        t_stitch = stitch()
        print("Stitching in {0:}".format(t_stitch))
    else:
        print("WARNING: at least one process didn't finish properly.")

if __name__ == '__main__':
    gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]
    iteration = 400000
    complete_inference(gpu_list, iteration)

