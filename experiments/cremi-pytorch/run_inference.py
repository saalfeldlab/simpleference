import vigra

import os
import sys
import time

from simpleference.inference.inference import run_inference_n5
# from simpleference.backends.pytorch import PyTorchPredict
from simpleference.backends.pytorch import InfernoPredict
from simpleference.backends.pytorch.preprocess import preprocess


def single_gpu_inference(sample, gpu):
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s_inference.n5' % sample
    model_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/networks/neurofire/mws/hed-1/Weights'
    out_file = '/groups/saalfeld/home/papec/Work/neurodata_hdd/networks/neurofire/mws/hed-1/Predictions/prediction_sample%s.n5' % sample
    assert os.path.exists(out_file)

    input_shape = (40, 396, 396)
    output_shape = (32, 320, 320)
    prediction = InfernoPredict(model_path, crop=output_shape, gpu=0)
    postprocess = None

    server_address = ('localhost', 9999)

    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     postprocess,
                     raw_path,
                     out_file,
                     server_address,
                     input_key='data',
                     target_keys='full_affs',
                     input_shape=input_shape,
                     output_shape=output_shape,
                     channel_order=[list(range(19))])
    t_predict = time.time() - t_predict

    with open(os.path.join(out_file, 't-inf_gpu%i.txt' % gpu), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))
    return True


if __name__ == '__main__':
    sample = sys.argv[1]
    gpu = int(sys.argv[2])
    single_gpu_inference(sample, gpu)
