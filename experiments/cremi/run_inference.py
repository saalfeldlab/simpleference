import os
import sys
import time
import json
from simpleference.inference.inference import run_inference
from simpleference.backends.gunpowder.caffe.backend import build_caffe_prediction
from simpleference.backends.gunpowder.preprocess import preprocess


def single_gpu_inference(sample, gpu, iteration):
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/cremi_warped_sample%s.h5' % sample
    save_folder = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/prediction_blocks_%s' % sample

    prototxt = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi/long_range_v2/long_range_unet.prototxt'
    weights  = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi/long_range_v2/net_iter_%i.caffemodel' % iteration

    offset_file = './offsets_sample%s/list_gpu_%i.json' % (sample, gpu)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    prediction = build_caffe_prediction(prototxt, weights, gpu)
    t_predict = time.time()
    run_inference(prediction, preprocess, raw_path, save_folder, offset_list)
    t_predict = time.time() - t_predict

    with open(os.path.join(save_folder, 't-inf_gpu%i.txt' % gpu), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    sample = sys.argv[1]
    gpu = int(sys.argv[2])
    iteration = int(sys.argv[3])
    single_gpu_inference(sample, gpu, iteration)
