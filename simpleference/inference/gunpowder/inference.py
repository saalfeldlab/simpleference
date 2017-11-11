import os
import h5py
from gunpowder import VolumeTypes

from ..backend.gunpowder.caffe.backend import CaffePredict
from ..backend.gunpowder.preprocess import preprocess


def build_caffe_prediction(prototxt, weights, gpu):
    assert os.path.exists(prototxt)
    assert os.path.exists(weights)

    pred = CaffePredict(prototxt,
                        weights,
                        inputs={
                            VolumeTypes.RAW: 'data'
                        },
                        outputs={
                            VolumeTypes.PRED_AFFINITIES: 'aff_pred'
                        },
                        use_gpu=gpu)
    return pred


# TODO
def build_tensorflow_prediction():
    pass


# TODO consider writing to a single big h5 or N5
def run_gunpowder_inference(prediction,
                            raw_path,
                            save_folder,
                            offset_list,
                            input_shape=(84, 268, 268)):

    assert callable(prediction)
    assert os.path.exists(raw_path)
    print("Starting prediction for data %s." % raw_path)
    n_blocks = len(offset_list)
    print("For %i number of blocks" % n_blocks)

    # prototxt = './long_range_unet.prototxt' if long_range else './default_unet.prototxt'
    # weights  = './net_iter_%i.caffemodel' % iteration

    for ii, off in enumerate(offset_list):
        print("Predicting block", ii, "/", len(offset_list))

        bb = tuple(slice(off[i], off[i] + input_shape[i]) for i in range(len(off)))
        with h5py.File(raw_path, 'r') as f:
            data = f['data'][bb]

        data = preprocess(data)

        out = prediction({'data': data})
        out = out['aff_pred']

        save_name = 'block_%s.h5' % '_'.join([str(o) for o in off])
        save_file = os.path.join(save_folder, save_name)
        with h5py.File(save_file, 'w') as f:
            f.create_dataset('data', data=out, compression='gzip')
