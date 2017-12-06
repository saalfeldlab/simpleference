from __future__ import print_function
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call

sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/nnets/simpleference')
from simpleference.inference.util import offset_list_from_precomputed
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld27/python')
import z5py

from precompute_offsets import precompute_offset_list


def single_inference(path, gpu, iteration):
    call(['./run_inference.sh', path, str(gpu), str(iteration)])
    return True


def complete_inference(path, gpu_list, iteration):

    assert os.path.exists(path), "Path to N5 dataset with raw data and mask does not exist"
    f = z5py.File(path, use_zarr_format=False)
    assert 'gray' in f, "Raw data not present in N5 dataset"
    assert 'mask' in f, "Mask not present in N5 dataset"

    shape = f['gray'].shape

    # create the datasets
    out_shape = (56,) *3

    # the n5 datasets might exist already
    if not 'affs_xy' in f:
        f.create_dataset('affs_xy', shape=shape,
                         compressor='gzip',
                         dtype='float32',
                         chunks=out_shape)
    if not 'affs_z' in f:
        f.create_dataset('affs_z', shape=shape,
                         compressor='gzip',
                         dtype='float32',
                         chunks=out_shape)

    # make the offset files, that assign blocks to gpus
    output_shape = (56, 56, 56)
    # generate offset lists with mask
    offset_list = precompute_offset_list(path, output_shape)
    mhash = hash(path)
    offset_list_from_precomputed(offset_list, gpu_list, './offsets_%i' % mhash)

    # run multiprocessed inference
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
        tasks = [pp.submit(single_inference, path, gpu, iteration) for gpu in gpu_list]
        result = [t.result() for t in tasks]

    if all(result):
        print("All gpu's finished inference properly.")
    else:
        print("WARNING: at least one process didn't finish properly.")


if __name__ == '__main__':
    gpu_list = list(range(8))
    iteration = 400000
    path = sys.argv[1]
    complete_inference(path, gpu_list, iteration)
