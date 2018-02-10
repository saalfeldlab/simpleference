from __future__ import print_function
import sys
import hashlib
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
    output_shape = (60, 596, 596)

    # the n5 datasets might exist already
    if 'predictions/full_affs' not in f:

        if 'predictions' not in f:
            f.create_group('predictions')

        chunks = (3,) + tuple(outs // 2 for outs in output_shape)
        aff_shape = (12,) + shape
        f.create_dataset('predictions/full_affs',
                         shape=aff_shape,
                         compression='gzip',
                         dtype='float32',
                         chunks=chunks)

    # make the offset files, that assign blocks to gpus
    # generate offset lists with mask
    offset_list = precompute_offset_list(path, output_shape)
    mhash = hashlib.md5(path.encode('utf-8')).hexdigest()
    offset_list_from_precomputed(offset_list, gpu_list, './offsets_%s' % mhash)

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
    paths = ['/nrs/saalfeld/lauritzen/01/workspace.n5/raw',
             '/nrs/saalfeld/lauritzen/02/workspace.n5/raw',
             '/nrs/saalfeld/lauritzen/02/workspace.n5/filtered',
             '/nrs/saalfeld/lauritzen/03/workspace.n5/raw',
             '/nrs/saalfeld/lauritzen/03/workspace.n5/filtered',
             '/nrs/saalfeld/lauritzen/04/workspace.n5/raw',
             '/nrs/saalfeld/lauritzen/04/workspace.n5/filtered'
            ]
    for path in paths:
        complete_inference(path, gpu_list, iteration)
