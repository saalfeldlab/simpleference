import h5py
import time
from simpleference.inference.util import stitch_prediction_blocks, extract_nn_affinities


def stitch(sample):
    save_folder = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/prediction_blocks_%s' % sample
    save_prefix = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/cremi_warped_sample%s_affinities' % sample

    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/cremi_warped_sample%s.h5' % sample

    with h5py.File(raw_path, 'r') as f:
        spatial_shape = f['data'].shape

    # n_channels = 12
    # shape = (n_channels,) + spatial_shape

    t_st = time.time()
    extract_nn_affinities(save_prefix, save_folder, spatial_shape)
    # stitch_prediction_blocks(save_file, save_folder, shape)
    t_st  = time.time() - t_st
    print("Stitching took %f s" % t_st)
    return t_st


if __name__ == '__main__':
    samples = ['A+', 'B+', 'C+']
    times = []
    for sample in samples:
        t_stitch = stitch(sample)
        times.append(t_stitch)

    for tt in times:
        print("Stitching in %f s" % tt)
