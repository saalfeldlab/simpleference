import sys
import numpy as np

sys.path.append('../../..')
from simpleference.masking import make_prediction_blocks, order_blocks
sys.path.append('/home/papec/Work/my_projects/cremi_tools')
from cremi_tools.viewer.volumina import view


def prediction_blocks(mask_path, output_shape, blocks_out):
    shape = (7062, 133718, 248156)
    downscale_factor = (13, 128, 128)
    mask = make_prediction_blocks(shape, downscale_factor, output_shape, mask_path, blocks_out)
    view([mask])


def make_ordered_blocks(blocks_in, blocks_out):
    shape = (7062, 133718, 248156)
    central_coordinate = np.array([sh / 2 for sh in shape])
    order_blocks(blocks_in, blocks_out, central_coordinate)


if __name__ == '__main__':
    # dummy output shape
    # output_shape = (56, 560, 560)
    # prediction_blocks('/home/papec/Work/neurodata_hdd/fafb/mask.h5',
    #                   output_shape,
    #                   './dummy-blocks.json')

    # order dummy blocks
    make_ordered_blocks('./dummy-blocks.json',
                        './dummy-blocks_ordered.json')
