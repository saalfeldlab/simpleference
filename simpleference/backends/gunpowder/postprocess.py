import numpy as np
import scipy.ndimage


def threshold_cc(data, thr=0.):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    output = (data > thr).astype(np.uint64)
    scipy.ndimage.label(output, output=output)
    return [data.squeeze(), output.squeeze()]


def nn_affs(data):
    output = np.empty(shape=(2,)+data.shape[1:], dtype=data.dtype)
    output[0] = (data[1] + data[2]) / 2.
    output[1] = data[0]
    return output
