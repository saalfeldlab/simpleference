from __future__ import print_function
import numpy as np
import scipy.ndimage


def threshold_cc(data, output_bounding_box, thr=0., ds_shape=(1, 1, 1), output_shape=(1, 1, 1)):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    output = (data > thr).astype(np.uint64)
    scipy.ndimage.label(output, output=output)
    block_offset = [int(output_bounding_box[k].start // output_shape[k]) for k in range(len(output_shape))]
    block_shape = [int(np.ceil(s/float(o))) for s, o in zip(ds_shape, output_shape)]
    print("Processing block at:", block_offset)
    id_offset = np.ravel_multi_index(block_offset, block_shape)*np.prod(output_shape)
    output[output > 0] += id_offset
    return [data.squeeze(), output.squeeze()]


def nn_affs(data, output_bounding_box):
    output = np.empty(shape=(2,)+data.shape[1:], dtype=data.dtype)
    output[0] = (data[1] + data[2]) / 2.
    output[1] = data[0]
    return output
