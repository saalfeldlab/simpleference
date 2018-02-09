import os
import sys
import time
import json
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess


def single_gpu_inference(sample, gpu, iteration):
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/n5/cremi_warped_sample%s.n5' % sample
    assert os.path.exists(raw_path), raw_path
    out_file = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/n5/cremi_warped_sample%s_predictions.n5' % sample
    assert os.path.exists(out_file), out_file

    meta_graph = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi-tf/unet_default/unet_checkpoint_%i' % iteration
    net_io_json = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi-tf/unet_default/net_io_names.json'
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    offset_file = './offsets_sample%s/list_gpu_%i.json' % (sample, gpu)
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
                     out_file,
                     offset_list,
                     input_shape=input_shape,
                     output_shape=output_shape,
                     only_nn_affs=True)
    t_predict = time.time() - t_predict

    with open(os.path.join(out_file, 't-inf_gpu%i.txt' % gpu), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    sample = sys.argv[1]
    gpu = int(sys.argv[2])
    iteration = int(sys.argv[3])
    single_gpu_inference(sample, gpu, iteration)
