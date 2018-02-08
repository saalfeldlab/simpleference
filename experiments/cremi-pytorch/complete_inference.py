from __future__ import print_function
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call

import z5py
from simpleference.inference.util import get_offset_lists


def single_inference(sample, gpu):
    call(['./run_inference.sh', sample, str(gpu)])
    return True


def complete_inference(sample,
                       gpu_list):

    # path to the raw data
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf['raw'].shape

    # create the datasets
    out_shape = (56,) * 3
    out_file = '/groups/saalfeld/home/papec/torch_master_test_sample%s.n5' % sample

    # the n5 file might exist already
    if not os.path.exists(out_file):
        f = z5py.File(out_file, use_zarr_format=False)
        f.create_dataset('affs_xy', shape=shape,
                         compressor='gzip',
                         dtype='float32',
                         chunks=out_shape)
        f.create_dataset('affs_z', shape=shape,
                         compressor='gzip',
                         dtype='float32',
                         chunks=out_shape)

    # make the offset files, that assign blocks to gpus
    save_folder = './offsets_sample%s' % sample
    output_shape = (56, 56, 56)
    get_offset_lists(shape, gpu_list, save_folder, output_shape=output_shape)

    # run multiprocessed inference
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
        tasks = [pp.submit(single_inference, sample, gpu) for gpu in gpu_list]
        result = [t.result() for t in tasks]

    if all(result):
        print("All gpu's finished inference properly.")
    else:
        print("WARNING: at least one process didn't finish properly.")


if __name__ == '__main__':
    for sample in ('A+',):
        gpu_list = [3, 4, 5, 6, 7]
        complete_inference(sample, gpu_list)
