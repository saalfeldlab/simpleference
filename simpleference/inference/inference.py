from __future__ import print_function
import os
import numpy as np

from .io import IoN5, IoDVID, IoHDF5


def load_input(io, offset, context, output_shape, padding_mode='reflect'):
    starts = [off - context[i] for i, off in enumerate(offset)]
    stops  = [off + output_shape[i] + context[i] for i, off in enumerate(offset)]
    shape = io.shape

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
    data = io.read(bb)

    # pad if necessary
    if pad_left is not None or pad_right is not None:
        pad_left = (0, 0, 0) if pad_left is None else pad_left
        pad_right = (0, 0, 0) if pad_right is None else pad_right
        pad_width = tuple((pl, pr) for pl, pr in zip(pad_left, pad_right))
        data = np.pad(data, pad_width, mode=padding_mode)

    return data


def run_inference_n5(prediction,
                     preprocess,
                     raw_path,
                     save_file,
                     offset_list,
                     input_shape,
                     output_shape,
                     input_key='data',
                     rejection_criterion=None,
                     padding_mode='reflect'):
    assert os.path.exists(raw_path)
    assert os.path.exists(save_file)
    io_in = IoN5(raw_path, [input_key])
    io_out = IoN5(save_file, ['affs_xy', 'affs_z'], save_only_nn_affs=True)
    run_inference(prediction, preprocess, io_in, io_out, offset_list,
                  input_shape, output_shape, rejection_criterion, padding_mode)


def run_inference(prediction,
                  preprocess,
                  io_in,
                  io_out,
                  offset_list,
                  input_shape,
                  output_shape,
                  rejection_criterion=None,
                  padding_mode='reflect'):

    assert callable(prediction)
    assert callable(preprocess)
    assert len(output_shape) == len(input_shape)

    if rejection_criterion is not None:
        assert callable(rejection_criterion)

    n_blocks = len(offset_list)
    print("Starting prediction...")
    print("For %i number of blocks" % n_blocks)

    # the additional context requested in the input
    context = np.array([input_shape[i] - output_shape[i] for i in range(len(input_shape))]) / 2
    context = context.astype('uint32')

    shape = io_in.shape

    # iterate over all the offsets, get the input data and predict
    for ii, offset in enumerate(offset_list):

        print("Predicting block", ii, "/", n_blocks)
        data = load_input(io_in, offset, context, output_shape, padding_mode=padding_mode)

        if rejection_criterion is not None:
            if rejection_criterion(data):
                print("Rejecting block", ii, "/", n_blocks)
                continue

        out = prediction(preprocess(data))

        # crop if necessary
        stops = [off + outs for off, outs in zip(offset, out.shape[1:])]

        if any(stop > dim_size for stop, dim_size in zip(stops, shape)):
            bb = (slice(None), ) + tuple(slice(0, dim_size - off if stop > dim_size else None)
                                         for stop, dim_size, off in zip(stops, shape, offset))
            # print("Remove Padding")
            # print(bb)
            out = out[bb]

        out_bb = tuple(slice(off, off + outs) for off, outs in zip(offset, output_shape))
        io_out.write(out, out_bb)
