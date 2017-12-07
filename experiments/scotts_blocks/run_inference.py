import os
import sys
import time
import json
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess


def single_gpu_inference(path, gpu, iteration):
    assert os.path.exists(path), path

    meta_graph = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi-tf/unet_default/unet_checkpoint_%i' % iteration
    net_io_json = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi-tf/unet_default/net_io_names.json'
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    mhash = hash(path)
    offset_file = './offsets_%i/list_gpu_%i.json' % (mhash, gpu)
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
                     path, path,
                     offset_list,
                     input_shape=input_shape,
                     output_shape=output_shape,
                     input_key='gray')
    t_predict = time.time() - t_predict

    with open(os.path.join(path, 't-inf_gpu%i.txt' % gpu), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    path = sys.argv[1]
    gpu = int(sys.argv[2])
    iteration = int(sys.argv[3])
    single_gpu_inference(path, gpu, iteration)
