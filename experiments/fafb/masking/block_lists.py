import sys
import numpy as np
import os
import json

sys.path.append('../../..')
from simpleference.masking import make_prediction_blocks, order_blocks


def prediction_blocks(mask_path, output_shape, blocks_out):
    shape = (7062, 133718, 248156)
    downscale_factor = (13, 128, 128)
    mask = make_prediction_blocks(shape, downscale_factor, output_shape, mask_path, blocks_out)

    # look at the mask shape to double check
    # sys.path.append('/home/papec/Work/my_projects/cremi_tools')
    # from cremi_tools.viewer.volumina import view
    # view([mask])


def make_ordered_blocks(blocks_in, blocks_out):
    shape = (7062, 133718, 248156)
    central_coordinate = np.array([sh / 2 for sh in shape])
    order_blocks(blocks_in, blocks_out, central_coordinate, resolution=(10, 1, 1))


def divide_list(block_file, blocks_out_first, blocks_out_second, first_part):
    with open(block_file, 'r') as f:
        blocks = json.load(f)
    split_block = int(round(len(blocks)*first_part))
    with open(blocks_out_first, 'w') as f:
        json.dump(blocks[:split_block], f)
    with open(blocks_out_second, 'w') as f:
        json.dump(blocks[split_block:], f)


if __name__ == '__main__':
    # dummy output shape
    output_shape = (71, 650, 650)
    metadata_folder = '/nrs/saalfeld/heinrichl/fafb_meta'
    #prediction_blocks('/groups/saalfeld/home/papec/Work/neurodata_hdd/fafb/masking/mask_s7.h5',
    #                  output_shape,
    #                  os.path.join(metadata_folder, 'block_list_in_mask.json'))
    # order dummy blocks
    #make_ordered_blocks(os.path.join(metadata_folder, 'block_list_in_mask.json'),
    #                    os.path.join(metadata_folder, 'block_list_in_mask_ordered.json'))
    divide_list(os.path.join(metadata_folder, 'block_list_in_mask_ordered_part2.json'),
                os.path.join(metadata_folder, 'block_list_in_mask_ordered_part2_local.json'),
                os.path.join(metadata_folder, 'block_list_in_mask_ordered_part2_lsf.json'), 1/9.)