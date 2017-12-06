import os
import sys
import time
import json
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/nnets/simpleference')
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess


def single_gpu_inference(gpu, iteration, gpu_offset):
    # path to the raw data
    raw_path = '/nrs/saalfeld/sample_E/sample_E.n5'
    path_in_file = 'volumes/raw/s0'

    save_file = '/groups/saalfeld/saalfeldlab/sampleE/affinity_predictions.n5'

    meta_graph = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi-tf/unet_default/unet_checkpoint_%i' % iteration
    net_io_json = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi-tf/unet_default/net_io_names.json'
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    offset_file = './offset_lists/list_gpu_%i.json' % (gpu + gpu_offset,)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    input_key = net_io_names["raw"]
    output_key = net_io_names["affs"]
    input_shape = (84, 268, 268)
    output_shape = (56, 56, 56)
    prediction = TensorflowPredict(meta_graph,
                                   input_key=input_key,
                                   output_key=output_key)
    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     raw_path,
                     save_file,
                     offset_list,
                     input_key=path_in_file,
                     input_shape=input_shape,
                     output_shape=output_shape)
    t_predict = time.time() - t_predict

    with open(os.path.join(save_file, 't-inf_gpu%i.txt' % (gpu + gpu_list,)), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    gpu = int(sys.argv[1])
    iteration = int(sys.argv[2])
    gpu_offset = int(sys.argv[3])
    single_gpu_inference(gpu, iteration, gpu_offset)
