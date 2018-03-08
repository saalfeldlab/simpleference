import numpy as np


def nn_affs(input_, output_bounding_box):
    output = np.empty(shape=(2,) + input_.shape[1:], dtype=input_.dtype)
    output[0] = (input_[1] + input_[2]) / 2.
    output[1] = input_[0]
    return output