from __future__ import print_function
import h5py
import os
import json
from concurrent import futures
from random import shuffle


# this returns the offsets for the given output blocks.
# blocks are padded on the fly in the inference if necessary
def get_offset_lists(path,
                     gpu_list,
                     save_folder,
                     output_shape=(56, 56, 56),
                     randomize=False):
    assert os.path.exists(path), path
    with h5py.File(path, 'r') as f:
        shape = f['data'].shape

    in_list = []
    for z in range(0, shape[0], output_shape[0]):
        for y in range(0, shape[1], output_shape[1]):
            for x in range(0, shape[2], output_shape[2]):
                in_list.append([z, y, x])

    if randomize:
        shuffle(in_list)

    n_splits = len(gpu_list)
    out_list = [in_list[i::n_splits] for i in range(n_splits)]

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for ii, olist in enumerate(out_list):
        list_name = os.path.join(save_folder, 'list_gpu_%i.json' % gpu_list[ii])
        with open(list_name, 'w') as f:
            json.dump(olist, f)


def stitch_prediction_blocks(save_path,
                             block_folder,
                             shape,
                             key = 'data',
                             end_channel=None,
                             n_workers=8,
                             chunks=(64, 64, 64)):

    if end_channel is None:
        chan_slice = (slice(None),)
    else:
        assert end_channel <= shape[0]
        chan_slice = (slice(0, end_channel),)

    def stitch_block(ds, block_id, block_file, n_blocks):
        print("Stitching block %i / %i" % (block_id, n_blocks))
        offsets = [int(off) for off in block_file[:-3].split('_')[1:]]
        with h5py.File(os.path.join(block_folder, block_file), 'r') as g:
            block_data = g['data'][:]
        block_shape = block_data.shape[1:]
        # Need to add slice for channel dimension
        bb = chan_slice + tuple(slice(off, off + block_shape[ii])
                                for ii, off in enumerate(offsets))
        ds[bb] = block_data


    with h5py.File(save_path, 'w') as f:
        ds = f.create_dataset(key,
                              shape=shape,
                              dtype='float32',
                              compression='gzip',
                              chunks=chunks)
        files = os.listdir(block_folder)
        # filter out invalid filenames
        files = [ff for ff in files if ff.startswith('block')]
        # make sure all blocks are h5 files
        assert all(ff[-3:] == '.h5' for ff in files)

        with futures.ThreadPoolExecutor(max_workers=n_workers) as tp:
            tasks = [tp.submit(stitch_block, ds, block_id, block_file, n_blocks)
                     for block_id, block_file in enumerate(files)]
            [t.result() for t in tasks]


def extract_nn_affinities(save_prefix,
                          block_folder,
                          shape,
                          invert_affs=False):

    save_path_xy = save_prefix + '_xy.h5'
    save_path_z = save_prefix + '_z.h5'
    with h5py.File(save_path_xy, 'w') as f_xy, h5py.File(save_path_z, 'w') as f_z:
        ds_xy = f_xy.create_dataset('data',
                                    shape=shape,
                                    dtype='float32',
                                    compression='gzip',
                                    chunks=(56, 56, 56))
        ds_z = f_z.create_dataset('data',
                                  shape=shape,
                                  dtype='float32',
                                  compression='gzip',
                                  chunks=(56, 56, 56))
        files = os.listdir(block_folder)

        def extract_block(i, ff):
            print("Stitching block %i / %i" % (i, len(files)))
            offsets = [int(off) for off in ff[:-3].split('_')[1:]]

            with h5py.File(os.path.join(block_folder, ff), 'r') as g:
                block_data = g['data'][:3]

            if invert_affs:
                block_data = 1. - block_data

            block_shape = block_data.shape[1:]
            # Need to add slice for channel dimension
            bb = tuple(slice(off, off + block_shape[ii]) for ii, off in enumerate(offsets))
            ds_xy[bb] = (block_data[1] + block_data[2]) / 2.
            ds_z[bb] = block_data[0]

        with futures.ThreadPoolExecutor(max_workers=20) as tp:
            tasks = []
            for i, ff in enumerate(files):
                if not ff.startswith('block'):
                    continue
                assert ff[-3:] == '.h5'
                tasks.append(tp.submit(extract_block, i, ff))
            [t.result() for t in tasks]


def reject_empty_batch(data):
    return np.sum(data) == 0
