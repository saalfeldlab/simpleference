import os
from math import floor, ceil
import json
import h5py
import numpy as np


def make_prediction_blocks(full_shape, downscale_factor, output_shape, mask_file, output_file):

    output_shape_ds = [outs / ds for outs, ds in zip(output_shape, downscale_factor)]

    pred_mask_shape = [sfull // outs + 1 for sfull, outs in zip(full_shape, output_shape)]

    assert os.path.exists(mask_file), mask_file
    with h5py.File(mask_file) as f:
        ds = f['data']
        mask = ds[:]

    print("Making prediction blocks for full shape", full_shape)
    print("Downscaled shape:", tuple(fs // ds for fs, ds in zip(full_shape, downscale_factor)))
    print("From mask shape :", mask.shape)
    print("(these should more or less agree !)")

    prediction_mask = np.zeros(pred_mask_shape, dtype='uint8')
    prediction_blocks = []
    # generate blocks
    for z in range(0, full_shape[0], output_shape[0]):
        print("generating for", z, "/", full_shape[0])
        for y in range(0, full_shape[1], output_shape[1]):
            for x in range(0, full_shape[2], output_shape[2]):
                z_ds = z / downscale_factor[0]
                y_ds = y / downscale_factor[1]
                x_ds = x / downscale_factor[2]

                stop_z = z_ds + output_shape_ds[0]
                stop_y = y_ds + output_shape_ds[1]
                stop_x = x_ds + output_shape_ds[2]

                bb = np.s_[
                    int(floor(z_ds)): int(ceil(stop_z)),
                    int(floor(y_ds)): int(ceil(stop_y)),
                    int(floor(x_ds)): int(ceil(stop_x))
                ]

                mask_block = mask[bb]

                if np.sum(mask_block) > 0:
                    prediction_blocks.append([z, y, x])
                    prediction_mask[z // output_shape[0], y // output_shape[1], x // output_shape[2]] = 1

    with open(output_file, 'w') as f:
        json.dump(prediction_blocks, f)

    n_blocks_total = np.prod([len(range(0, fs, os))
                              for fs, os in zip(full_shape, output_shape)])
    print("Percentage of blocks that will be predicted", len(prediction_blocks) / n_blocks_total)

    return prediction_mask


def order_blocks(block_file, out_file, central_coordinate):
    assert isinstance(central_coordinate, np.ndarray)
    assert len(central_coordinate) == 3
    with open(block_file, 'r') as f:
        prediction_blocks = np.array(json.load(f))

    distances = np.sum(np.square(prediction_blocks - central_coordinate), axis=1)
    sort = np.argsort(distances)

    prediction_blocks = prediction_blocks[sort]

    with open(out_file, 'w') as f:
        json.dump(prediction_blocks.tolist(), f)
