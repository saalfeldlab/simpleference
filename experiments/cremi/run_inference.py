import os
import sys
import time
import json
from simpleference.inference.gunpowder.inference import run_gunpowder_inference, build_caffe_prediction


def run_inference(sample, gpu, iteration):
    # TODO padded realigned volumes
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/mala_jan_original/raw/sample_%s.h5' % sample
    # TODO save somewhere reasoable
    save_folder = './prediction_blocks_%s' % sample

    # TODO proper paths
    prototxt = './long_range_unet.prototxt'
    weights  = './net_iter_%i.caffemodel' % iteration

    offset_file = './offsets_sample%s' % sample
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    prediction = build_caffe_prediction(prototxt, weights, gpu)
    t_predict = time.time()
    run_gunpowder_inference(prediction, raw_path, save_folder, offset_list)
    t_predict = time.time() - t_predict

    with open(os.path.join(save_folder, 't-inf_gpu%i.txt' % gpu), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    sample = sys.argv[1]
    gpu = int(sys.argv[2])
    iteration = int(sys.argv[3])
    run_inference()
