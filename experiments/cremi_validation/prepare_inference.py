from __future__ import  print_function
import sys
import os
import json

sys.path.append('/groups/saalfeld/home/heinrichl/Projects/simpleference')
from simpleference.inference.util import get_offset_lists
import z5py
import numpy as np
sys.path.append('/groups/saalfeld/home/heinrichl/Projects/simpleference')
from simpleference.inference.util import offset_list_from_precomputed
from lsf import script_generator_lsf
from precompute_offsets import precompute_offset_list


def run_filemaking(raw_path, out_shape, out_file, target_keys, data_key,config):
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf[data_key].shape
    # create the datasets
    assert os.path.exists(config['output_path'])
    if not os.path.exists(out_file):
        os.makedirs(out_file)

    for target_key in target_keys:
        f = z5py.File(out_file, use_zarr_format=False)
        split_keys = target_key.split('/')
        for group_key in split_keys[:-1]:
            if group_key not in f:
                f = f.create_group(group_key)
            else:
                f = f[group_key]
        ds = f.create_dataset(split_keys[-1],
                              shape=shape,
                              compression='gzip',
                              dtype='uint8',
                              chunks=out_shape)
        ds.attrs['inference_config']=config


def prepare_inference(blocklist_file, gpu_list, metadata_folder):
    offset_list_from_precomputed(blocklist_file, gpu_list, metadata_folder)
    script_generator_lsf.write_scripts(gpu_list)


def main(config_file):
    with open(config_file) as f:
        config = json.load(f)

    assert os.path.exists(config['raw_path']), "Path to N5 dataset with raw data and mask does not exist"
    assert os.path.exists(config['meta_path']), "Path to directory for meta data does not exist"
    precompute_offset_list(config['raw_path'],
                           config['output_shape'],
                           os.path.join(config['meta_path'],config['blocklist_file']),
                           mask_key=config['mask_keys'],
                           force_recomputation=config['force_recomputation'])
    run_filemaking(config['raw_path'],
                   config['output_shape'],
                   config['out_file'],
                   config['target_keys'],
                   config['data_key'],config)
    offset_list_from_precomputed(str(os.path.join(config['meta_path'], config['blocklist_file'])),
                                 config['gpu_list'],
                                 config['meta_path'],
                                 config['offset_list_name_extension'])
    script_generator_lsf.write_scripts(config['gpu_list'])


if __name__ == '__main__':
    experiment_names = ['DTU2_Bonly']
    #experiment_names = [
    #    'baseline_DTU2',
    #    'DTU2_unbalanced',
    #    'DTU2-small',
    #    'DTU2_100tanh',
    #    'DTU2_150tanh',
    #    'DTU2_Aonly',
    #    #'DTU2_Bonly',
    #    'DTU2_Conly',
    #    'DTU2_Adouble',
    #    'baseline_DTU1',
    #    'DTU1_unbalanced',
    #    'DTU2_plus_bdy',
    #    'DTU1_plus_bdy',
    #    'BCU2',
    #    'BCU1'
    #]
    for experiment_name in experiment_names:
        for sample in ['A', 'B', 'C']:
            for iteration in range(42000,56000,2000):
                config_file = 'config_{0:}_{1:}_{2:}.json'.format(experiment_name, sample, iteration)
                main(config_file)