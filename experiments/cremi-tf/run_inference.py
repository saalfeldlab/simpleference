import os
import sys
import time
import json
from simpleference.postprocessing import float_to_uint8
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess


def single_gpu_inference(sample, gpu, iteration):
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    assert os.path.exists(raw_path), raw_path
    out_file = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    assert os.path.exists(out_file), out_file

    net_folder = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi-tf'
    graph_weights = os.path.join(net_folder, 'unet_default/unet_checkpoint_%i' % iteration)
    graph_inference = os.path.join(net_folder, 'unet_default/unet_inference')
    net_io_json = os.path.join(net_folder, 'unet_default/net_io_names.json')
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    offset_file = './offsets_sample%s/list_gpu_%i.json' % (sample, gpu)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    input_key = net_io_names["raw"]
    output_key = net_io_names["affs"]
    input_shape = (88, 808, 808)
    output_shape = (60, 596, 596)
    prediction = TensorflowPredict(graph_weights,
                                   graph_inference,
                                   input_key=input_key,
                                   output_key=output_key)
    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     float_to_uint8,
                     raw_path,
                     out_file,
                     offset_list,
                     input_key='raw',
                     target_keys='predictions/full_affs',
                     input_shape=input_shape,
                     output_shape=output_shape,
                     channel_order=[list(range(12))])
    t_predict = time.time() - t_predict

    with open(os.path.join(out_file, 't-inf_gpu%i.txt' % gpu), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    sample = sys.argv[1]
    gpu = int(sys.argv[2])
    iteration = int(sys.argv[3])
    single_gpu_inference(sample, gpu, iteration)
