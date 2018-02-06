import os
import sys
import time
import json

from os.path import expanduser as user

from simpleference.inference.inference import run_inference_n5
from simpleference.backends.pytorch import PyTorchPredict
from simpleference.backends.pytorch.preprocess import preprocess


def single_gpu_inference(sample, gpu):
    raw_path = user('~/data/cremi_sample%s.n5' % sample)
    out_file = user('~/data/sample%s_affinities_pytorch_test.n5' % sample)

    model_path = user('~/data/model.pytorch')

    offset_file = './offsets_sample%s/list_gpu_%i.json' % (sample, gpu)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    input_shape = (84, 270, 270)
    output_shape = (56, 56, 56)
    prediction = PyTorchPredict(model_path,
                                crop=output_shape)

    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     raw_path,
                     out_file,
                     offset_list,
                     input_shape=input_shape,
                     output_shape=output_shape)
    t_predict = time.time() - t_predict

    with open(os.path.join(out_file, 't-inf_gpu%i.txt' % gpu), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    sample = sys.argv[1]
    gpu = int(sys.argv[2])
    single_gpu_inference(sample, gpu)
