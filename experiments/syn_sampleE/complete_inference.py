from __future__ import print_function
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call

sys.path.append('/groups/saalfeld/home/heinrichl/Projects/simpleference')
from simpleference.inference.util import offset_list_from_precomputed
import z5py

from precompute_offsets import precompute_offset_list


def single_inference(path, gpu, iteration):
    call(['./run_inference.sh', path, str(gpu), str(iteration)])
    return True


def complete_inference(gpu_list, iteration):
    path = '/nrs/saalfeld/sample_E/sample_E.n5'
    assert os.path.exists(path), "Path to N5 dataset with raw data and mask does not exist"
    rf = z5py.File(path, use_zarr_format=False)
    assert 'volumes/raw/s0' in rf, "Raw data not present in N5 dataset"

    shape = rf['volumes/raw/s0'].shape

    # create the datasets
    output_shape = (71, 650, 650)
    out_file = '/data/heinrichl/sample_E.n5'
    if not os.path.exists(out_file):
        os.mkdir(out_file)

    f = z5py.File(out_file, use_zarr_format=False)
    # the n5 datasets might exist already
    key = 'syncleft_dist_DTU-2_{0:}'.format(iteration)
    if key not in f:
        f.create_dataset(key,
                         shape=shape,
                         compression='gzip',
                         dtype='float32',
                         chunks=(71, 325, 325))


    # make the offset files, that assign blocks to gpus
    # generate offset lists with mask
    offset_folder = '/nrs/saalfeld/heinrichl/synapses/sampleE_DTU2_offsets_update/'
    if not os.path.exists(offset_folder):
        os.mkdir(offset_folder)
    offset_list = os.path.join(offset_folder, 'block_list_in_mask.json')
    offset_list_from_precomputed(offset_list, gpu_list, os.path.join(offset_folder))

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
    complete_inference(gpu_list, iteration)
