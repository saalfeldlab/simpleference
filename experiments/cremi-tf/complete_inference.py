from __future__ import print_function
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call

from simpleference.inference.util import get_offset_lists
import z5py


def single_inference(sample, gpu, iteration):
    call(['./run_inference.sh', sample, str(gpu), str(iteration)])
    return True


def complete_inference(sample,
                       gpu_list,
                       iteration):

    # path to the raw data
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    assert os.path.exists(raw_path), raw_path
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf['raw'].shape

    # create the datasets
    output_shape = (60, 596, 596)
    out_file = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample

    # the n5 file might exist already
    f = z5py.File(out_file, use_zarr_format=False)

    if 'predictions' not in f:
        f.create_group('predictions')

    if 'predictions/full_affs' not in f:
        chunks = (3,) + tuple(outs // 2 for outs in output_shape)
        aff_shape = (12,) + shape
        f.create_dataset('predictions/full_affs',
                         shape=aff_shape,
                         compression='gzip',
                         dtype='uint8',
                         chunks=chunks)

    # make the offset files, that assign blocks to gpus
    save_folder = './offsets_sample%s' % sample
    get_offset_lists(shape, gpu_list, save_folder, output_shape=output_shape)

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
    iteration = 400000
    # for sample in ('A+', 'B+', 'C+'):
    for sample in ('B', 'C'):
        complete_inference(sample, gpu_list, iteration)
