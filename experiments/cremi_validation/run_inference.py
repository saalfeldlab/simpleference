import os
import sys
import time
import json
import z5py
from functools import partial
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess
from simpleference.backends.gunpowder.postprocess import clip_float32_to_uint8, clip_float32_to_uint8_range_0_1


def single_gpu_inference(config, gpu):

    offset_file = os.path.join(config['meta_path'], 'list_gpu_{0:}{1:}.json'.format(gpu,
                                                                                    config[
                                                                                        'offset_list_name_extension']))
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)
    print(config['input_key'], config['output_key'])
    prediction = TensorflowPredict(config['weight_meta_graph']
                                   , config['inference_meta_graph'], input_key=config['input_key'],
                                   output_key=config['output_key'])
    if 'postprocess' in config:
        if config['postprocess'] == 'clip_float32_to_uint8_range_0_1':
            postprocess = clip_float32_to_uint8_range_0_1
        elif config['postprocess'] == 'clip_float32_to_uint8' or config['postprocess'] == \
                'clip_float32_to_uint8_range': #todo remove this dirty bug hack fix
            postprocess = clip_float32_to_uint8
    else:
        postprocess = clip_float32_to_uint8
    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     postprocess,
                     config['raw_path'],
                     config['out_file'],
                     offset_list,
                     input_key=config['data_key'],
                     input_shape=config['input_shape'],
                     output_shape=config['output_shape'],
                     target_keys=config['target_keys'],
                     log_processed=os.path.join(os.path.dirname(offset_file),
                                                'list_gpu_{0:}{1:}_processed.txt'.format(gpu,
                                                                                         config[
                                                                                             'offset_list_name_extension']))
                     )

    t_predict = time.time() - t_predict

    with open(os.path.join(os.path.dirname(offset_file), 't-inf_gpu{0:}.txt'.format(gpu)), 'w') as f:
        f.write("Inference with gpu {0:} in {1:} s\n".format(gpu, t_predict))


if __name__ == '__main__':
    gpu = int(sys.argv[1])


    # making a list of all configs:
    all_configs = []
    experiment_names = [
        'baseline_DTU2',
        'DTU2_unbalanced',
        'DTU2-small',
        'DTU2_100tanh',
        'DTU2_150tanh',
        'DTU2_Aonly',
        'DTU2_Conly',
        'DTU2_Adouble',
        'baseline_DTU1',
        'DTU1_unbalanced',
        'DTU2_plus_bdy',
        'DTU1_plus_bdy',
        'BCU2',
        'BCU1'
    ]
    for experiment_name in experiment_names:
        for sample in ['A', 'B', 'C']:
            for iteration in [2000, 4000, 6000, 8000, 12000, 14000,16000,18000,22000,24000,26000,28000,32000,34000,
                              36000, 38000]+list(range(42000,70000,2000)):
                config_file = 'config_{0:}_{1:}_{2:}.json'.format(experiment_name, sample, iteration)
                all_configs.append(config_file)
    experiment_name = 'DTU2_Bonly'
    for sample in ['A', 'B', 'C']:
        for iteration in [2000, 4000, 6000, 8000, 12000, 14000, 16000, 18000, 22000, 24000, 26000, 28000, 32000, 34000,
                          36000, 38000]:
            config_file = 'config_{0:}_{1:}_{2:}.json'.format(experiment_name, sample, iteration)
            all_configs.append(config_file)
    for experiment_name in experiment_names:
        for sample in ['A', 'B', 'C']:
            for iteration in range(70000,84000,2000):
                config_file = 'config_{0:}_{1:}_{2:}.json'.format(experiment_name, sample, iteration)
                all_configs.append(config_file)
    experiment_name = 'DTU2_Bonly'
    for sample in ['A', 'B', 'C']:
        for iteration in range(42000,56000,2000):
            config_file = 'config_{0:}_{1:}_{2:}.json'.format(experiment_name, sample, iteration)
            all_configs.append(config_file)

    # find the last config that has been processed
    my_gpu_has_processed_json = '{0:}_processed_configs.json'.format(gpu)
    if os.path.exists(my_gpu_has_processed_json):
        with open(my_gpu_has_processed_json, 'r') as f:
            my_gpu_has_processed = json.load(f)
        last_processed = my_gpu_has_processed[-1]
        print('I found the last processed config to be', last_processed)
        restart_here = all_configs.index(last_processed) + 1
    else:
        my_gpu_has_processed = []
        restart_here = 0
    for config_file in all_configs[restart_here:]:
        print(config_file)
        with open(config_file) as f:
             config = json.load(f)
        single_gpu_inference(config, gpu)
        my_gpu_has_processed.append(config_file)
        with open(my_gpu_has_processed_json, 'w') as f:
            json.dump(my_gpu_has_processed, f)
