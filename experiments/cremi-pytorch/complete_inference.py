from __future__ import print_function
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call

from simpleference.inference.util import get_offset_lists
import z5py


def single_inference(sample, gpu):
    call(['./run_inference.sh', sample, str(gpu)])
    return True


def complete_inference(sample, gpu_list):

    # path to the raw data
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s_inference.n5' % sample
    assert os.path.exists(raw_path), raw_path
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf['data'].shape

    # create the datasets
    output_shape = (32, 320, 320)
    out_file = '/groups/saalfeld/home/papec/Work/neurodata_hdd/networks/neurofire/mws/hed-1/Predictions/prediction_sample%s.n5' % sample

    # the n5 file might exist already
    f = z5py.File(out_file, use_zarr_format=False)

    if 'full_affs' not in f:
        # chunks = (3,) + tuple(outs // 2 for outs in output_shape)
        chunks = (3,) + output_shape
        aff_shape = (19,) + shape
        f.create_dataset('full_affs', shape=aff_shape,
                         compression='gzip', dtype='float32', chunks=chunks)

    # run multiprocessed inference
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
        tasks = [pp.submit(single_inference, sample, gpu) for gpu in gpu_list]
        result = [t.result() for t in tasks]

    if all(result):
        print("All gpu's finished inference properly.")
    else:
        print("WARNING: at least one process didn't finish properly.")


if __name__ == '__main__':
    gpu_list = list(range(8))
    samples = ('A+', 'B+', 'C+')
    sample = samples[2]
    complete_inference(sample, gpu_list)
