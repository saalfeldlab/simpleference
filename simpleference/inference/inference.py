from __future__ import print_function
import os

import numpy as np
import dask
import toolz as tz
import functools

from .io import IoN5  # , IoDVID, IoHDF5


def load_input(io, offset, context, output_shape, padding_mode='reflect'):
    starts = [off - context[i] for i, off in enumerate(offset)]
    stops = [off + output_shape[i] + context[i] for i, off in enumerate(offset)]
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


def run_inference_n5(predicters,
                     preprocess,
                     raw_path,
                     save_file,
                     offset_list,
                     input_shape,
                     output_shape,
                     input_key='data',
                     padding_mode='reflect',
                     only_nn_affs=False,
                     num_workers_per_gpu=4):

    assert os.path.exists(raw_path)
    assert os.path.exists(raw_path)
    assert os.path.exists(save_file)
    # The N5 IO/Wrapper needs iterables as keys
    # so we wrap the input key in a list.
    # Note that this is not the case for the hdf5 wrapper,
    # which just takes a single key.
    io_in = IoN5(raw_path, [input_key])
    # This is specific to the N5 datasets, where I have implemented
    # averaging over nearest neighbor xy-affinities and z affinities
    # seperately.
    keys = ['affs_xy', 'affs_z'] if only_nn_affs else ['full_affs']
    io_out = IoN5(save_file, keys, save_only_nn_affs=only_nn_affs)
    run_inference(predicters,
                  preprocess,
                  io_in,
                  io_out,
                  offset_list,
                  input_shape,
                  output_shape,
                  num_workers_per_gpu=num_workers_per_gpu,
                  padding_mode=padding_mode)
    # This is not necessary for n5 datasets
    # which do not need to be closed, but we leave it here for
    # reference when using other (hdf5) io wrappers
    io_in.close()
    io_out.close()


def run_inference(predicters,
                  preprocess,
                  io_in,
                  io_out,
                  offset_list,
                  input_shape,
                  output_shape,
                  num_workers_per_gpu=4,
                  padding_mode='reflect'):

    assert isinstance(predicters, dict)
    assert all(callable(predicter) for predicter in predicters.values())
    assert callable(preprocess)
    assert len(output_shape) == len(input_shape)

    gpu_list = list(predicters.keys())
    assert all(isinstance(gpu, int) for gpu in gpu_list)
    n_gpus = len(gpu_list)

    n_blocks = len(offset_list)
    print("Starting prediction...")
    print("For %i number of blocks" % n_blocks)

    # the additional context requested in the input
    context = np.array([input_shape[i] - output_shape[i]
                        for i in range(len(input_shape))]) / 2
    context = context.astype('uint32')

    shape = io_in.shape

    @dask.delayed
    def load_offset(offset):
        return load_input(io_in, offset, context, output_shape,
                          padding_mode=padding_mode)

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
        return output, output_bounding_box

    @dask.delayed
    def write_output(output, output_bounding_box):
        io_out.write(output, output_bounding_box)
        return 1
    preprocess = dask.delayed(preprocess)

    delayed_predicters = {gpu: dask.delayed(predicters[gpu]) for gpu in gpu_list}
    get = functools.partial(dask.threaded.get, num_workers=num_workers_per_gpu)

    @dask.delayed
    def infer_offset(offset, gpu, i):

        print("Predicting block ", i, "/", len(offset_list), "on gpu", gpu)
        predict = delayed_predicters[gpu]
        output = tz.pipe(offset, load_offset, preprocess, predict)
        output_crop, output_bounding_box = verify_shape(offset, output)
        result = write_output(output_crop, output_bounding_box)
        # NOTE: Because dask.compute doesn't take an argument, but rather an
        # arbitrary number of arguments, computing each in turn, the output of
        # dask.compute(results) is a tuple of length 1, with its only element
        # being the results list. If instead we pass the results list as *args,
        # we get the desired container of results at the end.
        success = dask.compute(result, get=get)
        return success

    # We want to parallelize inference in the following way:
    # We have a list of offsets, that is mapped to the pool of gpus (each gpu has its own `predicter` instance)
    # which process these offsets / tasks in parallel.
    # each task itself can spawn `num_workers_per_gpu` threads to feed the gpu
    tasks = [infer_offset(offset, gpu_list[ii % n_gpus], ii) for ii, offset in enumerate(offset_list)]
    result = dask.compute(*tasks, traverse=False,
                          get=dask.threaded.get, num_workers=len(gpu_list))
    print(f'Ran {sum(result)} jobs')
