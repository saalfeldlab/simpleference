import h5py
import time
from simpleference.inference.util import stitch_prediction_blocks


def stitch(sample):
    folder = './prediction_blocks_%s' % sample
    save_file = './prediction_sample_%s.h5' % sample

    # TODO padded realigned volumes
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/mala_jan_original/raw/sample_%s.h5' % sample

    with h5py.File(raw_path, 'r') as f:
        spatial_shape = f['data'].shape

    n_channels = 12
    shape = (n_channels,) + spatial_shape

    t_st = time.time()
    stitch_prediction_blocks(save_file, folder, shape)
    print("Stitching took %f s" % (time.time - t_st, ))


if __name__ == '__main__':
    sample = 'A+'
    stitch(sample)
