from __future__ import print_function
import os
import h5py
import numpy as np


def load_input(ds, offset, context, output_shape, padding_mode='reflect'):
    starts = [off - context[i] for i, off in enumerate(offset)]
    stops  = [off + output_shape[i] + context[i] for i, off in enumerate(offset)]
    shape = ds.shape

    # we pad the input volume if necessary
    pad_left = None
    pad_right = None

    # check for padding to the left
    if any(start < 0 for start in starts):
        pad_left = tuple(abs(start) if start < 0 else 0 for start in starts)
        starts = [max(0, start) for start in starts]

    # check for padding to the right
    if any(stop > shape[i] for i, stop in enumerate(stops)):
        pad_right = tuple(stop - shape[i] if stop > shape[i] else 0 for i, stop in enumerate(stops))
        stops = [min(shape[i], stop) for i, stop in enumerate(stops)]

    bb = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    data = ds[bb]

    # pad if necessary
    if pad_left is not None or pad_right is not None:
        pad_left = (0, 0, 0) if pad_left is None else pad_left
        pad_right = (0, 0, 0) if pad_right is None else pad_right
        pad_width = tuple((pl, pr) for pl, pr in zip(pad_left, pad_right))
        # TODO should we use constant padding with zeros instead of reflection padding ?
        data = np.pad(data, pad_width, mode=padding_mode)

    return data


# TODO consider writing to a single big h5 or N5
def run_inference(prediction,
                  preprocess,
                  raw_path,
                  save_folder,
                  offset_list,
                  output_shape=(56, 56, 56),
                  input_shape=(84, 268, 268),
                  rejection_criterion=None,
                  padding_mode='reflect'):

    assert callable(prediction)
    assert callable(preprocess)
    assert os.path.exists(raw_path)
    assert len(output_shape) == len(input_shape)

    if rejection_criterion is not None:
        assert callable(rejection_criterion)

    n_blocks = len(offset_list)
    print("Starting prediction for data %s." % raw_path)
    print("For %i number of blocks" % n_blocks)

    # the additional context requested in the input
    context = np.array([input_shape[i] - output_shape[i] for i in range(len(input_shape))]) / 2
    context = context.astype('uint32')

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    with h5py.File(raw_path, 'r') as f:
        # TODO we shouldn't hardcode keys...
        ds = f['data']
        shape = ds.shape

        # iterate over all the offsets, get the input data and predict
        for ii, offset in enumerate(offset_list):

            print("Predicting block", ii, "/", n_blocks)
            data = load_input(ds, offset, context, output_shape, padding_mode=padding_mode)

            if rejection_criterion is not None:
                if rejection_criterion(data):
                    continue

            out = prediction(preprocess(data))

            # crop if necessary
            stops = [off + outs for off, outs in zip(offset, out.shape[1:])]

            if any(stop > dim_size for stop, dim_size in zip(stops, shape)):
                bb = (slice(None), ) + tuple(slice(0, dim_size - off if stop > dim_size else None)
                                             for stop, dim_size, off in zip(stops, shape, offset))
                out = out[bb]

            save_name = 'block_%s.h5' % '_'.join([str(off) for off in offset])
            save_file = os.path.join(save_folder, save_name)
            with h5py.File(save_file, 'w') as g:
                g.create_dataset('data', data=out, compression='gzip')
