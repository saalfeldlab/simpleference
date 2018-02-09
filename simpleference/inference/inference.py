from __future__ import print_function
import os
import z5py
import h5py
import numpy as np
import dask
import toolz as tz
import functools
from shutil import rmtree


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
                  input_shape,
                  output_shape,
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
                    print("Rejecting block", ii, "/", n_blocks)
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


def run_inference_n5(prediction,
                     preprocess,
                     raw_path,
                     save_file,
                     offset_list,
                     input_shape,
                     output_shape,
                     input_key='data',
                     rejection_criterion=None,
                     padding_mode='reflect',
                     num_cpus=2):

    assert os.path.exists(raw_path)
    assert len(output_shape) == len(input_shape)

    n_blocks = len(offset_list)
    print("Starting prediction for data %s." % raw_path)
    print("For %i number of blocks" % n_blocks)

    # the additional context requested in the input
    context = np.array([input_shape[i] - output_shape[i]
                        for i in range(len(input_shape))]) / 2
    context = context.astype('uint32')

    # create out file and read in file
    f = z5py.File(raw_path, use_zarr_format=False)
    ds = f[input_key]
    shape = ds.shape

    print("Writing prediction to %s." % save_file)
    g = z5py.File(save_file, use_zarr_format=False)
    ds_xy = g['affs_xy']
    ds_z = g['affs_z']

    @dask.delayed
    def load_offset(offset):
        print('loading offset %s' % offset)
        return load_input(ds, offset, context, output_shape,
                          padding_mode=padding_mode)
    preprocess = dask.delayed(preprocess)
    predict = dask.delayed(prediction)

    @dask.delayed(nout=2)
    def verify_shape(offset, output):
        # crop if necessary
        stops = [off + outs for off, outs in zip(offset, output.shape[1:])]

        if any(stop > dim_size for stop, dim_size in zip(stops, shape)):
            bb = ((slice(None),) +
                  tuple(slice(0, dim_size - off if stop > dim_size else None)
                        for stop, dim_size, off in zip(stops, shape, offset)))
            output = output[bb]

        output_bounding_box = tuple(slice(off, off + outs)
                                    for off, outs in zip(offset, output_shape))
        print('done verifying shape')
        return output, output_bounding_box

    @dask.delayed
    def write_output(output, output_bounding_box):
        print('writing output at %s' % [(s.start, s.stop) for s in
                                        output_bounding_box])
        ds_xy[output_bounding_box] = (output[1] + output[2]) / 2.
        ds_z[output_bounding_box] = output[0]
        return 1

    # iterate over all the offsets, get the input data and predict
    results = []
    for offset in offset_list:
        output = tz.pipe(offset, load_offset, preprocess, predict)
        output_crop, output_bounding_box = verify_shape(offset, output)
        result = write_output(output_crop, output_bounding_box)
        results.append(result)

    get = functools.partial(dask.threaded.get, num_workers=num_cpus)
    print(f'We have {len(results)} tasks waiting to complete.')
    # NOTE: Because dask.compute doesn't take an argument, but rather an
    # arbitrary number of arguments, computing each in turn, the output of
    # dask.compute(results) is a tuple of length 1, with its only element
    # being the results list. If instead we pass the results list as *args,
    # we get the desired container of results at the end.
    success = dask.compute(*results, get=get)
    print(f'Ran {sum(success)} jobs')
