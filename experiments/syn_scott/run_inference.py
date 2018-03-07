import os
import sys
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld27/python')

import time
import json
import z5py
from functools import partial
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess
from simpleference.postprocessing import *

def single_gpu_inference(sample, gpu, iteration):
    path = '/nrs/saalfeld/lauritzen/%s/workspace.n5/raw' % sample
    assert os.path.exists(path), path
    rf = z5py.File(path, use_zarr_format=False)
    shape = rf['gray'].shape
    weight_meta_graph = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/unet_checkpoint_%i' % iteration
    inference_meta_graph = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/unet_inference'
    net_io_json = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/net_io_names.json'

    out_file = '/nrs/saalfeld/heinrichl/test/lauritzen/%s/workspace.n5' % sample
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    mhash = hash(path)
    offset_file = '/nrs/saalfeld/heinrichl/synapses/scott_offsets_{0:}_DTU2_inf/offsets_{' \
                  '1:}/list_gpu_{2:}.json'.format(sample, mhash, gpu)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    input_key = net_io_names["raw"]
    output_key = net_io_names["dist"]
    input_shape = (91, 862, 862)
    output_shape = (71, 650, 650)
    prediction = TensorflowPredict(weight_meta_graph,
                                   inference_meta_graph,
                                   input_key=input_key,
                                   output_key=output_key)
    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     partial(threshold_cc, thr=0., output_shape=output_shape, ds_shape=shape),
                     path,
                     out_file,
                     offset_list,
                     input_shape=input_shape,
                     output_shape=output_shape,
                     target_keys=('syncleft_dist_DTU-2_{0:}'.format(iteration),'syncleft_cc_DTU-2_{0:}'.format(
                         iteration)),
                     input_key='gray',
                     log_processed=os.path.join(os.path.dirname(offset_file), 'list_gpu_{0:}_{'
                                                                                '1:}_processed.txt'.format(gpu,
                                                                                                           iteration)))
    t_predict = time.time() - t_predict

    with open(os.path.join(os.path.dirname(offset_file), 't-inf_gpu_{0:}_{1:}.txt'.format(gpu, iteration)), 'w') as f:
        f.write("Inference with gpu %i in %f s\n" % (gpu, t_predict))


if __name__ == '__main__':
    sample = sys.argv[1]
    gpu = int(sys.argv[2])
    iteration = int(sys.argv[3])
    single_gpu_inference(sample, gpu, iteration)
