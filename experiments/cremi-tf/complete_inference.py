from __future__ import print_function
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call

sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/nnets/simpleference')
from simpleference.inference.util import get_offset_lists
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld/python')
import z5py


def single_inference(sample, gpu, iteration):
    call(['./run_inference.sh', sample, str(gpu), str(iteration)])
    return True


def complete_inference(sample,
                       gpu_list,
                       iteration):

    # path to the raw data
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/n5/cremi_warped_sample%s.n5' % sample
    assert os.path.exists(raw_path), raw_path
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf['data'].shape

    # create the datasets
    out_shape = (56,) *3
    out_file = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/n5/cremi_warped_sample%s_predictions.n5' % sample

    # the n5 file might exist already
    f = z5py.File(out_file, use_zarr_format=False)
    if not 'affs_xy' in f:
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
    get_offset_lists(shape, gpu_list, save_folder, output_shape=out_shape)

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
    # gpu_list = [0]
    iteration = 400000
    for sample in ('B+', 'C+'):
        complete_inference(sample, gpu_list, iteration)
