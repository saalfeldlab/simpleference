import os
import logging
import z5py
from butler import start_service
from butler.block_service import BlockService, BlockRequestHandler
from simpleference.inference.util import get_offset_lists


def start_block_service(sample):

    # make the offset files, that assign blocks to gpus
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s_inference.n5' % sample
    assert os.path.exists(raw_path), raw_path
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf['data'].shape
    output_shape = (32, 320, 320)
    save_folder = './offsets_sample%s' % sample
    get_offset_lists(shape, [0], save_folder, output_shape=output_shape)

    host = 'localhost'
    port = 9999
    logging.basicConfig(level=logging.INFO)
    service = BlockService('./offsets_sample%s/list_gpu_0.json' % sample, 120)
    start_service(host, port, service, BlockRequestHandler)


if __name__ == '__main__':
    start_block_service('C+')
