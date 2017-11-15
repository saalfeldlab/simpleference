from __future__ import print_function
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call
from simpleference.inference.util import get_offset_lists


def single_inference(sample, gpu, iteration):
    call(['./run_inference.sh', sample, str(gpu), str(iteration)])
    return True


def complete_inference(sample, gpu_list, iteration):

    # path to the raw data
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/cremi_warped_sample%s.h5' % sample

    # make the offset files, that assign blocks to gpus
    save_folder = './offsets_sample%s' % sample
    output_shape = (56, 56, 56)
    get_offset_lists(raw_path, gpu_list, save_folder, output_shape=output_shape)

    # run multiprocessed inference
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
        tasks = [pp.submit(single_inference, sample, gpu, iteration) for gpu in gpu_list]
        result = [t.result() for t in tasks]

    if all(result):
        print("All gpu's finished inference properly.")
    else:
        print("WARNING: at least one process didn't finish properly.")


if __name__ == '__main__':
    gpu_list = [0, 2, 3, 4, 5, 6, 7]
    iteration = 100000
    for sample in ('B+', 'C+'):
        complete_inference(sample, gpu_list, iteration)
