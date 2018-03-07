from __future__ import print_function
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call

sys.path.append('/groups/saalfeld/home/heinrichl/Projects/simpleference')
from simpleference.inference.util import offset_list_from_precomputed
import z5py

from precompute_offsets import precompute_offset_list


def single_inference(sample, gpu, iteration):
    call(['./run_inference.sh', sample, str(gpu), str(iteration)])
    return True


def complete_inference(sample, gpu_list, iteration):
    path = '/nrs/saalfeld/lauritzen/%s/workspace.n5/raw' % sample
    assert os.path.exists(path), "Path to N5 dataset with raw data and mask does not exist"
    rf = z5py.File(path, use_zarr_format=False)
    assert 'gray' in rf, "Raw data not present in N5 dataset"
    assert 'mask' in rf, "Mask not present in N5 dataset"

    shape = rf['gray'].shape

    # create the datasets
    output_shape = (71, 650, 650)
    out_file = '/nrs/saalfeld/heinrichl/test/lauritzen/%s/workspace.n5' %sample
    if not os.path.exists(out_file):
        os.mkdir(out_file)

    f = z5py.File(out_file, use_zarr_format=False)
    # the n5 datasets might exist already

    f.create_dataset('syncleft_dist_DTU-2_{0:}'.format(iteration),
                     shape=shape,
                     compressor='gzip',
                     dtype='float32',
                     chunks=output_shape)
    f.create_dataset('syncleft_cc_DTU-2_{0:}'.format(iteration),
                     shape=shape,
                     compressor='gzip',
                     dtype='uint64',
                     chunks=output_shape)


    # make the offset files, that assign blocks to gpus
    # generate offset lists with mask
    offset_folder = '/nrs/saalfeld/heinrichl/synapses/scott_offsets_{0:}_DTU2_inf/'.format(sample)
    if not os.path.exists(offset_folder):
        os.mkdir(offset_folder)
    offset_list = precompute_offset_list(path, output_shape, offset_folder)
    mhash = hash(path)
    offset_list_from_precomputed(offset_list, gpu_list, os.path.join(offset_folder, 'offsets_%i' % mhash))

    # run multiprocessed inference
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
        tasks = [pp.submit(single_inference, sample, gpu, iteration) for gpu in gpu_list]
        result = [t.result() for t in tasks]

    if all(result):
        print("All gpu's finished inference properly.")
    else:
        print("WARNING: at least one process didn't finish properly.")


if __name__ == '__main__':
    gpu_list = list(range(8))
    iteration = 550000
    for sample in ['01', '02', '03', '04']:
        complete_inference(sample, gpu_list, iteration)
