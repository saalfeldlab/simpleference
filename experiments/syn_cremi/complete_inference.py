from __future__ import  print_function
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call

sys.path.append('/groups/saalfeld/home/heinrichl/Projects/simpleference')
from simpleference.inference.util import get_offset_lists
import z5py


def single_inference(sample, gpu, iteration):
    call(['./run_inference.sh', sample, str(gpu), str(iteration)])
    return True


def complete_inference(sample, gpu_list, iteration):
    # path to the raw data
    raw_path = '/groups/saalfeld/saalfeldlab/larissa/data/cremi/cremi_warped_sample{0:}.n5/volumes/'.format(sample)
    assert os.path.exists(raw_path), "Path to N5 dataset with raw data and mask does not exist"
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf['raw'].shape

    # create the datasets
    out_shape = (71, 650, 650)
    out_file = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/'
    if not os.path.exists(out_file):
        os.mkdir(out_file)
    out_file = os.path.join(out_file,
                            'prediction_cremi_warped_sample{0:}_{1:}.n5'.format(sample, iteration))

    f = z5py.File(out_file, use_zarr_format=False)
    f.create_dataset('syncleft_dist',
                     shape=shape,
                     compressor='gzip',
                     dtype='float32',
                     chunks=out_shape)
    f.create_dataset('syncleft_cc',
                     shape=shape,
                     compressor='gzip',
                     dtype='uint64',
                     chunks=out_shape)

    # make the offset files, that assign blocks to gpus
    offset_folder = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/offsets_{0:}_{1:}_x{2:}_y{3:}_z{4:}/'.format(
        sample, iteration, out_shape[0], out_shape[1], out_shape[2])
    if not os.path.exists(offset_folder):
        os.mkdir(offset_folder)

    get_offset_lists(shape, gpu_list, offset_folder, output_shape=out_shape)

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
    iteration = 448000
    for sample in ["A+", "B+", "C+"]:
        complete_inference(sample, gpu_list, iteration)
