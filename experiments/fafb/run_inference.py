from __future__ import print_function
import os
import sys
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld27/python')

import time
import json
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess
from simpleference.postprocessing import *
from functools import partial


def curate_processed_list(file_to_list):
    with open(file_to_list, 'r') as f:
        list_as_str = f.read()
    curated_list_as_str = list_as_str[:list_as_str.rfind(']')+1]
    file_to_curated_list = os.path.splitext(file_to_list)[0]+'_curated.json'
    with open(file_to_curated_list, 'w') as f:
        f.write('['+curated_list_as_str+']')
    return file_to_curated_list


def single_gpu_inference(gpu, iteration, list_extension=''):
    raw_path = '/groups/saalfeld/saalfeldlab/FAFB00/v14_align_tps_20170818_dmg.n5/volumes/raw/'
    assert os.path.exists(raw_path), raw_path
    weight_meta_graph = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/unet_checkpoint_%i' % iteration
    inference_meta_graph = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/unet_inference'
    net_io_json = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/net_io_names.json'
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    out_file = '/nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5/volumes/predictions/synapses_dt'
    metadata_folder = '/nrs/saalfeld/heinrichl/fafb_meta/'
    offset_file = os.path.join(metadata_folder, 'list_gpu_{0:}{1:}.json'.format(gpu, list_extension))
    processed_file = os.path.join(metadata_folder, 'list_gpu_{0:}_processed{1:}.txt'.format(gpu, list_extension))

    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    if os.path.exists(processed_file):
        curated_processed_file = curate_processed_list(processed_file)
        with open(curated_processed_file, 'r') as f:
            processed_list = json.load(f)[:-1]
            processed_list_set = set(tuple(coo) for coo in processed_list)
    else:
        processed_list_set = set()

    offset_list_set = set(tuple(coo) for coo in offset_list)

    if processed_list_set == offset_list_set:
        print("processing was complete")
        return
    assert processed_list_set < offset_list_set
    offset_list = [list(coo) for coo in offset_list_set-processed_list_set]
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
                     partial(clip_float_to_uint8, float_range=(-1, 1), safe_scale=False),
                     raw_path,
                     out_file,
                     offset_list,
                     input_shape=input_shape,
                     output_shape=output_shape,
                     target_keys='s0',
                     input_key='s0',
                     log_processed=processed_file)
    t_predict = time.time() - t_predict

    with open(os.path.join(metadata_folder, 't-inf_gpu_{0:}{1:}.txt'.format(gpu, list_extension)), 'w') as f:
        f.write("Inference with gpu %i in %f s\n" % (gpu, t_predict))


if __name__ == '__main__':
    gpu = int(sys.argv[1])
    iteration = 550000
    list_extension = '_part2_missing'

    single_gpu_inference(gpu, iteration, list_extension=list_extension)

