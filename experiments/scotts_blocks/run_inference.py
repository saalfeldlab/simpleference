import os
import sys
import time
import json
import hashlib
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess


def single_gpu_inference(path, gpu, iteration):
    assert os.path.exists(path), path

    net_folder = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi-tf'
    graph_weights = os.path.join(net_folder, 'unet_default/unet_checkpoint_%i' % iteration)
    graph_inference = os.path.join(net_folder, 'unet_default/unet_inference')
    net_io_json = os.path.join(net_folder, 'unet_default/net_io_names.json')
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)


    mhash = hashlib.md5(path.encode('utf-8')).hexdigest()
    offset_file = './offsets_%s/list_gpu_%i.json' % (mhash, gpu)
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
                     path, path,
                     offset_list,
                     input_shape=input_shape,
                     output_shape=output_shape,
                     input_key='gray',
                     target_keys='predictions/full_affs',
                     full_affinities=True)
    t_predict = time.time() - t_predict

    with open(os.path.join(path, 't-inf_gpu%i.txt' % gpu), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    path = sys.argv[1]
    gpu = int(sys.argv[2])
    iteration = int(sys.argv[3])
    single_gpu_inference(path, gpu, iteration)
