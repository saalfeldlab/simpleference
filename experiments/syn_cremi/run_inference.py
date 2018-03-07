import os
import sys
import time
import json
import z5py
from functools import partial
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess
from simpleference.postprocessing import *


def single_gpu_inference(sample, gpu, iteration):
    raw_path = '/groups/saalfeld/saalfeldlab/larissa/data/cremi/cremi_warped_sample{0:}.n5/volumes'.format(sample)
    assert os.path.exists(raw_path), "Path to N5 dataset with raw data and mask does not exist"
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf['raw'].shape

    weight_meta_graph = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/unet_checkpoint_{0:}'.format(iteration)
    inference_meta_graph = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/unet_inference'
    net_io_json = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/net_io_names.json'

    out_file = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/'
    out_file = os.path.join(out_file, 'prediction_cremi_warped_sample{0:}_{1:}.n5'.format(sample, iteration))
    assert os.path.exists(out_file)

    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    input_key = net_io_names["raw"]
    output_key = net_io_names["dist"]
    input_shape = (91, 862, 862)
    output_shape = (71, 650, 650)

    offset_file = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/' \
                  'offsets_{0:}_{1:}_x{2:}_y{3:}_z{4:}/list_gpu_{5:}.json'.format(sample, iteration, output_shape[0],
                                                                             output_shape[1], output_shape[2], gpu)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)
    prediction = TensorflowPredict(weight_meta_graph, inference_meta_graph, input_key=input_key, output_key=output_key)

    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     clip_float_to_uint8,
                     raw_path,
                     out_file,
                     offset_list,
                     input_key='raw',
                     input_shape=input_shape,
                     output_shape=output_shape,
                     target_keys=('syncleft_dist'),
                     log_processed=os.path.join(os.path.dirname(offset_file),
                                                'list_gpu_{0:}_processed.txt'.format(gpu))
                     )

    t_predict = time.time() - t_predict

    with open(os.path.join(os.path.dirname(offset_file), 't-inf_gpu{0:}.txt'.format(gpu)), 'w') as f:
        f.write("Inference with gpu {0:} in {1:} s\n".format(gpu, t_predict))


if __name__ == '__main__':
    sample = sys.argv[1]
    gpu = int(sys.argv[2])
    iteration = int(sys.argv[3])
    single_gpu_inference(sample, gpu, iteration)