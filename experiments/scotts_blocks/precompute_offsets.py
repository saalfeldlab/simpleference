import sys
import json
import os
from itertools import chain
from concurrent import futures
import numpy as np
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld27/python')
import z5py


def precompute_offset_list(path, output_shape, n_threads=40):
    mhash = hash(path)
    tmp_path = 'precomputed_list_%i.json' % mhash
    # load the list if we have already computed it
    if os.path.exists(tmp_path):
        with open(tmp_path, 'r') as f:
            return json.load(f)

    # otherwise, compute it
    f = z5py.File(path, use_zarr_format=False)
    ds = f['mask']
    shape = ds.shape

    def generate_blocks(z, y, x):
        stop_z = min(z + output_shape[0], shape[0])
        stop_y = min(y + output_shape[1], shape[1])
        stop_x = min(x + output_shape[2], shape[2])

        bb = np.s_[z:stop_z, y:stop_y, z:stop_z]
        mask = ds[bb]

        # we run prediction for this block if any of the pixels is not masked
        if np.sum(mask) > 0:
            return [z, y, x]

    with futures.ThreadPoolExecutor(max_workers=n_threads) as tp:
        tasks = [tp.submit(generate_blocks, z, y, x) for z in range(0, shape[0], output_shape[0])
                                                     for y in range(0, shape[1], output_shape[1])
                                                     for x in range(0, shape[2], output_shape[2])]
        prediction_blocks = [t.result() for t in tasks]
    prediction_blocks = [pred_block for pred_block in prediction_blocks if pred_block is not None]

    with open(tmp_path, 'w') as blockfile:
        json.dump(prediction_blocks, blockfile)

    return prediction_blocks
