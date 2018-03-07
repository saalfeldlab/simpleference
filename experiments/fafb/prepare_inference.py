import os
import sys
sys.path.append('.')
import z5py
import numpy as np
sys.path.append('/groups/saalfeld/home/heinrichl/Projects/simpleference')
from simpleference.inference.util import offset_list_from_precomputed

from lsf import script_generator_lsf
from local import script_generator_local


def prepare_inference(gpu_list):
    # path to the raw data
    raw_path = '/groups/saalfeld/saalfeldlab/FAFB00/v14_align_tps_20170818_dmg.n5/volumes/raw'
    assert os.path.exists(raw_path), "Path to N5 dataset with raw data and mask does not exist"
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf['s0'].shape

    # create the datasets
    out_shape = (71, 650, 650)
    out_file = '/nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5'
    if not os.path.exists(out_file):
        os.mkdir(out_file)
    f = z5py.File(out_file, use_zarr_format=False)
    g1 = f.create_group('volumes')
    g2 = g1.create_group('predictions')
    g3 = g2.create_group('synapses_dt')
    g3.create_dataset('s0',
                      shape=shape,
                      compression='gzip',
                      level=6,
                      dtype='uint8',
                      chunks=out_shape)

    metadata_folder = '/nrs/saalfeld/heinrichl/fafb_meta/'
    assert os.path.exists(metadata_folder)
    offset_list_from_precomputed(os.path.join(metadata_folder, 'list_gpu_all_part2_missing.json'),
                                 gpu_list,
                                 metadata_folder, list_name_extension='_part2_missing')
    #offset_list_from_precomputed(os.path.join(metadata_folder, 'block_list_in_mask_ordered_part2_local.json'),
    #                             gpu_list_local, metadata_folder, list_name_extension='_part2')
    script_generator_lsf.write_scripts(gpu_list)
    script_generator_local.write_scripts(gpu_list, list(range(8))*(len(gpu_list)/8)+list(range(len(
        gpu_list)%8)))
    script_generator_local.write_scripts_choose_gpu(gpu_list)


if __name__ == '__main__':
    #gpu_list_local = list(range(16))
    gpu_list = list(range(52))
    prepare_inference(gpu_list)