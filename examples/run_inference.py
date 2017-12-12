import os
import sys
import time
import json
import z5py

from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess
from simpleference.inference.util import get_offset_lists


def single_gpu_inference(raw_path,
                         out_path,
                         net_folder,
                         gpu,
                         iteration):

    assert os.path.exists(raw_path)
    assert os.path.exists(net_folder)
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf['data'].shape
    input_shape = (84, 268, 268)
    output_shape = (56, 56, 56)

    # the n5 file might exist already
    if not os.path.exists(out_path):
        f = z5py.File(out_path, use_zarr_format=False)
        f.create_dataset('affs_xy', shape=shape,
                         compressor='gzip',
                         dtype='float32',
                         chunks=output_shape)
        f.create_dataset('affs_z', shape=shape,
                         compressor='gzip',
                         dtype='float32',
                         chunks=output_shape)

    # make offset list
    get_offset_lists(shape, [gpu], './offsets', output_shape=output_shape)

    meta_graph = os.path.join(net_folder, 'unet_checkpoint_%i' % iteration)
    net_io_json = os.path.join(net_folder, 'net_io_names.json')
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    offset_file = './offsets/list_gpu_%i.json' % gpu
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    input_key = net_io_names["raw"]
    output_key = net_io_names["affs"]
    prediction = TensorflowPredict(meta_graph,
                                   input_key=input_key,
                                   output_key=output_key)
    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     raw_path,
                     out_path,
                     offset_list,
                     input_shape=input_shape,
                     output_shape=output_shape)
    t_predict = time.time() - t_predict
    print("Running inference in %f s" % t_predict)



if __name__ == '__main__':
    raw_path = sys.argv[1]
    out_path = sys.argv[2]
    net_folder = sys.argv[3]
    gpu = int(sys.argv[4])
    iteration = int(sys.argv[5])
    single_gpu_inference(raw_path, out_path, net_folder, gpu, iteration)
