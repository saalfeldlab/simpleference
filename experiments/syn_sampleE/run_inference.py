import os
import sys
import time
import json
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess


def single_gpu_inference(path, gpu, iteration):

    assert os.path.exists(path), path
    out_file = '/data/heinrichl/sample_E.n5'
    weight_meta_graph = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/unet_checkpoint_%i' % iteration
    inference_meta_graph = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/unet_inference'
    net_io_json = '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/net_io_names.json'

    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    offset_file = '/nrs/saalfeld/heinrichl/synapses/sampleE_DTU2_offsets_update/list_gpu_{0:}.json'.format(gpu)
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
                     path,
                     out_file,
                     offset_list,
                     input_shape=input_shape,
                     output_shape=output_shape,
                     target_keys='syncleft_dist_DTU-2_{0:}'.format(iteration),
                     input_key='volumes/raw/s0',
                     log_processed = '/nrs/saalfeld/heinrichl/synapses/sampleE_DTU2_offsets_update/list_gpu_{'
                                     '0:}_processed.txt'.format(gpu)
    )
    t_predict = time.time() - t_predict

    with open(os.path.join(out_file, 't-inf_gpu%i.txt' % gpu), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    sample = sys.argv[1]
    gpu = int(sys.argv[2])
    iteration = int(sys.argv[3])
    single_gpu_inference(sample, gpu, iteration)
