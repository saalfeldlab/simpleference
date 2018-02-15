import sys
import vigra
import numpy as np

sys.path.append('../../..')
from simpleference.masking import postprocess_ilastik_predictions
sys.path.append('/home/papec/Work/my_projects/cremi_tools')
from cremi_tools.viewer.volumina import view


# the channel mapping we have used so far:
# channel 0: black
# channel 1: resin
# channel 2: data
def postprocess_mask(prediction_path, raw_path, out_path):

    # get rid of axis tags
    prediction = vigra.readHDF5(prediction_path, 'exported_data').view(np.ndarray)
    # TODO transpose ?!
    prediction = prediction.transpose((2, 1, 0, 3))

    mask = postprocess_ilastik_predictions(prediction[..., 2])

    print("Percentage in mask:", np.sum(mask) / mask.size)

    # save mask
    vigra.writeHDF5(mask, out_path, 'data', compression='gzip')

    # I screwed that one up...
    raw = vigra.readHDF5(raw_path, 'key')
    view([raw, prediction, mask])


def extract_scale_leve(scale):
    assert scale >= 6
    sys.path.append('/home/papec/Work/my_projects/z5/bld/python')
    import z5py
    path = '/home/papec/mnt/saalfeldlab/FAFB00/v14_align_tps_20170818_dmg.n5/volumes/raw'
    key = 's%i' % scale
    data = z5py.File(path, use_zarr_format=False)[key][:]
    out_path = '/home/papec/Work/neurodata_hdd/fafb/raw.h5'
    vigra.writeHDF5(data, out_path, key, compression='gzip', chunks=(64, 64, 64))


if __name__ == '__main__':
    # extract_scale_leve(7)
    postprocess_mask(
        '/home/papec/Work/neurodata_hdd/fafb/raw_Probabilities.h5',
        '/home/papec/Work/neurodata_hdd/fafb/raw.h5',
        '/home/papec/Work/neurodata_hdd/fafb/mask.h5')
