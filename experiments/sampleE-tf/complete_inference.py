from __future__ import print_function
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call

import z5py


def single_inference(gpu, iteration, gpu_offset):
    print(gpu, iteration, gpu_offset)
    call(['./run_inference.sh', str(gpu), str(iteration), str(gpu_offset)])
    return True


def complete_inference(gpu_list, iteration, gpu_offset):

    # create the datasets
    output_shape = (60, 596, 596)

    raw_path = '/nrs/saalfeld/sample_E/sample_E.n5'
    g = z5py.File(raw_path)
    shape = g['volumes/raw/s0'].shape

    # open the datasets
    save_path = '/nrs/saalfeld/sample_E/sample_E.n5'
    f = z5py.File(save_path, use_zarr_format=False)
    if 'volumes/predictions/full_affs' not in f:
        chunks = tuple(outs // 2 for outs in output_shape)
        chunks = (3,) + chunks
        f.create_dataset('volumes/predictions/full_affs',
                         shape=(12,) + shape,
                         compression='gzip',
                         level=6,
                         dtype='uint8',
                         chunks=chunks)

    # run multiprocessed inference
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
        tasks = [pp.submit(single_inference, gpu, iteration, gpu_offset) for gpu in gpu_list]
        result = [t.result() for t in tasks]

    if all(result):
        print("All gpu's finished inference properly.")
    else:
        print("WARNING: at least one process didn't finish properly.")


if __name__ == '__main__':
    gpu_list = range(8)
    iteration = 400000
    gpu_offset = int(sys.argv[1])
    complete_inference(gpu_list, iteration, gpu_offset)
