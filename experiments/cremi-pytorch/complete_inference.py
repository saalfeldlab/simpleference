from __future__ import print_function
import os
import json

import z5py

from simpleference.inference.util import get_offset_lists
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.pytorch import PyTorchPredict
from simpleference.backends.pytorch.preprocess import preprocess


def complete_inference(sample, gpu_list):

    # path to the raw data
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf['raw'].shape

    # create the datasets
    input_shape = (84, 270, 270)
    output_shape = (56, 56, 56)
    out_file = '/groups/saalfeld/home/papec/torch_master_test_sample%s.n5' % sample

    # the n5 file might exist already
    if not os.path.exists(out_file):
        f = z5py.File(out_file, use_zarr_format=False)
        f.create_dataset('affs_xy', shape=shape,
                         compressor='gzip',
                         dtype='float32',
                         chunks=output_shape)
        f.create_dataset('affs_z', shape=shape,
                         compressor='gzip',
                         dtype='float32',
                         chunks=output_shape)

    # make the offset files, that assign blocks to gpus
    save_folder = './offsets_sample%s' % sample
    output_shape = (56, 56, 56)
    # FIXME dirty hack to get single offset list for a single gpu
    get_offset_lists(shape, 1, save_folder, output_shape=output_shape)
    offset_file = './offsets_sample%s/list_gpu_%i.json' % (sample, 0)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    model_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/networks/neurofire'
    model_path = os.path.join(model_path, 'criteria_exps/sorensen_dice_unweighted/Weights/networks/model.pytorch')
    predicters = {gpu: PyTorchPredict(model_path, crop=output_shape, gpu=gpu) for gpu in gpu_list}
    run_inference_n5(predicters, preprocess,
                     raw_path, out_file, offset_list,
                     input_shape=input_shape,
                     output_shape=output_shape,
                     input_key='raw',
                     only_nn_affs=True)


if __name__ == '__main__':
    for sample in ('A+',):
        gpu_list = list(range(8))
        complete_inference(sample, gpu_list)
