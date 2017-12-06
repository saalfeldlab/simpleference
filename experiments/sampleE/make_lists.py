import os
from simpleference.inference.util import offset_list_from_precomputed


def make_list():
    in_list = './block_list_in_mask.json'
    gpu_list = range(16)
    offset_list_from_precomputed(in_list, gpu_list, './offset_lists')


if __name__ == '__main__':
    make_list()
