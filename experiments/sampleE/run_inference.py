import os
import sys
import time
import json
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.caffe.backend import CaffePredict
from simpleference.backends.gunpowder.preprocess import preprocess


def single_gpu_inference(gpu, iteration, gpu_offset):
    # path to the raw data
    raw_path = '/nrs/saalfeld/sample_E/sample_E.n5/volumes/raw/s0'

    save_file = '/groups/saalfeld/saalfeldlab/sampleE/my_prediction.n5'
    if not os.path.exists(os.path.split(save_file)[0]):
        os.mkdir(os.path.split(save_file)[0])

    prototxt = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi/long_range_v2/long_range_unet.prototxt'
    weights  = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi/long_range_v2/net_iter_%i.caffemodel' % iteration

    offset_file = './offset_lists/list_gpu_%i.json' % (gpu + gpu_offset,)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    input_key = 'data'
    output_key = 'aff_pred'
    input_shape = (84, 268, 268)
    output_shape = (56, 56, 56)
    prediction = CaffePredict(prototxt,
                              weights,
                              gpu=gpu,
                              input_key=input_key,
                              output_key=output_key)
    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     raw_path,
                     save_file,
                     offset_list,
                     input_shape=input_shape,
                     output_shape=output_shape)
    t_predict = time.time() - t_predict

    with open(os.path.join(save_folder, 't-inf_gpu%i.txt' % (gpu + gpu_list,)), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    gpu = int(sys.argv[1])
    iteration = int(sys.argv[2])
    gpu_offset = int(sys.argv[3])
    single_gpu_inference(gpu, iteration, gpu_offset)
