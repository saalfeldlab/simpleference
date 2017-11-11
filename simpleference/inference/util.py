from __future__ import print_function
import h5py
import os
import json


# this returns the offsets for the given output blocks.
# blocks are padded on the fly in the inference if necessary
def get_offset_lists(path,
                     gpu_list,
                     save_folder,
                     output_shape=(56, 56, 56)):
    assert os.path.exists(path), path
    with h5py.File(path, 'r') as f:
        shape = f['data'].shape

    in_list = []
    for z in range(0, shape[0], output_shape[0]):
        for y in range(0, shape[1], output_shape[1]):
            for x in range(0, shape[2], output_shape[2]):
                in_list.append([z, y, x])

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
                             shape):

    with h5py.File(save_path, 'w') as f:
        ds = f.create_dataset('data', shape=shape, dtype='float32', compression='gzip')
        files = os.listdir(block_folder)

        for i, ff in enumerate(files):

            print("Stitching block %i / %i" % (i, len(files)))

            if not ff.startswith('block'):
                continue

            assert ff[-3:] == '.h5'
            offsets = [int(off) for off in ff[:-3].split('_')[1:]]

            with h5py.File(os.path.join(block_folder, ff), 'r') as g:
                block_data = g['data'][:]

            block_shape = block_data.shape[1:]
            # Need to add slice for channel dimension
            bb = (slice(None),) + tuple(slice(off, off + block_shape[ii])
                                        for ii, off in enumerate(offsets))
            ds[bb] = block_data
