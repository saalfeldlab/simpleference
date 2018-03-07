import sys
import json
import os
import hashlib
from itertools import chain
from concurrent import futures
import numpy as np
import warnings
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld/python')
import z5py


def precompute_offset_list(path, output_shape, blocklist_file='precomputed_list.json', n_threads=40,
                           mask_key='mask', force_recomputation=False):

    # load the list if we have already computed it
    if os.path.exists(blocklist_file) and not force_recomputation:
        with open(blocklist_file, 'r') as f:
            warnings.warn("The file already existed, loading content from existing file. To force recomputation, "
                          "set force_recomputation=True")
            return json.load(f)

    # otherwise, compute it
    f = z5py.File(path, use_zarr_format=False)
    print(path, mask_key)
    if isinstance(mask_key, str):
        mask_key=(mask_key,)
    ds = list()
    for mk in mask_key:
        ds.append(f[mk])

    shape = ds[0].shape
    print("Precomputing offset list for volume with shape", shape)

    def generate_blocks(z, y, x):
        stop_z = min(z + output_shape[0], shape[0])
        stop_y = min(y + output_shape[1], shape[1])
        stop_x = min(x + output_shape[2], shape[2])

        bb = np.s_[z:stop_z, y:stop_y, x:stop_x]
        mask = np.sum(d[bb] for d in ds)

        # we run prediction for this block if any of the pixels is not masked
        if np.sum(mask) > 0:
            return [z, y, x]

    with futures.ThreadPoolExecutor(max_workers=n_threads) as tp:
        tasks = [tp.submit(generate_blocks, z, y, x) for z in range(0, shape[0], output_shape[0])
                                                     for y in range(0, shape[1], output_shape[1])
                                                     for x in range(0, shape[2], output_shape[2])]
        prediction_blocks = [t.result() for t in tasks]
    prediction_blocks = [pred_block for pred_block in prediction_blocks if pred_block is not None]
    print("Found %i valid blocks" % len(prediction_blocks))

    with open(blocklist_file, 'w') as blockfile:
        json.dump(prediction_blocks, blockfile)

    return prediction_blocks
