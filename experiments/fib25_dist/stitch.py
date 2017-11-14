import h5py
import time
from simpleference.inference.util import stitch_prediction_blocks


def stitch():
    save_folder = '/nrs/saalfeld/heinrichl/segmentation/distance_thirdtest/fib25_sub_prediction_at_400000/'
    save_path = '/nrs/saalfeld/heinrichl/segmentation/distance_thirdtest/fib25_sub_prediction_at_400000.h5'
    raw_path = '/groups/saalfeld/saalfeldlab/larissa/data/fib25/grayscale_sub.h5'

    with h5py.File(raw_path, 'r') as f:
        spatial_shape = f['data'].shape

    n_channels = 1
    shape = (n_channels,) + spatial_shape

    t_st = time.time()
    #extract_nn_affinities(save_prefix, save_folder, spatial_shape)
    stitch_prediction_blocks(save_path, save_folder, shape)
    t_st  = time.time() - t_st
    print("Stitching took %f s" % t_st)
    return t_st


if __name__ == '__main__':
    t_stitch = stitch()
    print("Stitching in {0:}".format(t_stitch))
