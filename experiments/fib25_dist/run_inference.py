import os
import sys
import time
import json
from simpleference.inference.inference import run_inference
from simpleference.inference.util import reject_empty_batch
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess


def single_gpu_inference(gpu, iteration):
    raw_path = '/groups/saalfeld/saalfeldlab/larissa/data/fib25/grayscale_sub.h5'
    save_folder = '/nrs/saalfeld/heinrichl/segmentation/distance_thirdtest/fib25_sub_prediction_at_%i' % iteration

    meta_graph = '/nrs/saalfeld/heinrichl/segmentation/distance_thirdtest/unet_checkpoint_%i' % iteration
    net_io_json = '/nrs/saalfeld/heinrichl/segmentation/distance_thirdtest/net_io_names.json'
    offset_file = '/nrs/saalfeld/heinrichl/segmentation/distance_thirdtest/offsets/list_gpu_%i.json' % gpu
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    prediction = TensorflowPredict(meta_graph, net_io_names['raw'],net_io_names['dist'])
    t_predict = time.time()
    run_inference(prediction, preprocess, raw_path, save_folder, offset_list, output_shape=(44,)*3, input_shape=(132,
                                                                                                                 )*3,
                  rejection_criterion=reject_empty_batch, padding_mode='constant')
    t_predict = time.time() - t_predict

    with open(os.path.join(save_folder, 't-inf_gpu%i.txt' % gpu), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))
    TensorflowPredict.stop()
if __name__ == '__main__':

    gpu = int(sys.argv[1])
    iteration = int(sys.argv[2])
    single_gpu_inference(gpu, iteration)
