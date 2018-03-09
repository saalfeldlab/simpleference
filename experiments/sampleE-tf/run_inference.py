import os
import sys
import time
import json
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess
from simpleference.postprocessing import float_to_uint8


def single_gpu_inference(gpu, iteration, gpu_offset):
    # path to the raw data
    raw_path = '/nrs/saalfeld/sample_E/sample_E.n5'
    path_in_file = 'volumes/raw/s0'

    save_path = '/nrs/saalfeld/sample_E/sample_E.n5'

    net_folder = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi-tf'
    graph_weights = os.path.join(net_folder, 'unet_default/unet_checkpoint_%i' % iteration)
    graph_inference = os.path.join(net_folder, 'unet_default/unet_inference')
    net_io_json = os.path.join(net_folder, 'unet_default/net_io_names.json')
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    offset_file = './offset_lists/list_gpu_%i.json' % (gpu + gpu_offset,)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    input_shape = (88, 808, 808)
    output_shape = (60, 596, 596)

    input_key = net_io_names["raw"]
    output_key = net_io_names["affs"]
    prediction = TensorflowPredict(graph_weights,
                                   graph_inference,
                                   input_key=input_key,
                                   output_key=output_key)
    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     float_to_uint8,
                     raw_path,
                     save_path,
                     offset_list,
                     input_shape,
                     output_shape,
                     input_key=path_in_file,
                     target_keys=['volumes/predictions/full_affs'],
                     num_cpus=10,
                     channel_order=[list(range(12))],
                     log_processed='./processed_gpu_%i.txt' % (gpu + gpu_offset))

    t_predict = time.time() - t_predict

    with open(os.path.join(save_path, 't-inf_gpu%i.txt' % (gpu + gpu_offset,)), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    gpu = int(sys.argv[1])
    iteration = int(sys.argv[2])
    gpu_offset = int(sys.argv[3])
    single_gpu_inference(gpu, iteration, gpu_offset)
