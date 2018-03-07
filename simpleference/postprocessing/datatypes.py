import numpy as np


def float_to_uint8(input_):
    return np.round(input_ * 255.).astype('uint8')


# TODO put Larissa's converter here
def float_to_int8(input_):
    pass
