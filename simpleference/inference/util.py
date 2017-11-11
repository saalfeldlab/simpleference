import h5py
import os
import json


def get_offset_lists(path,
                     gpu_list,
                     save_folder,
                     input_shape=(84, 268, 268),
                     output_shape=(56, 56, 56)):
    assert os.path.exists(path)
    with h5py.File(path, 'r') as f:
        shape = f['data'].shape

    offset = tuple((input_shape[i] - output_shape[i]) // 2
                   for i in range(3))

    in_list = []
    for z in range(0, shape[0], output_shape[0]):
        for y in range(0, shape[1], output_shape[1]):
            for x in range(0, shape[2], output_shape[2]):
                offsets = [z, y, x]
                if any([off + input_shape[i] >= shape[i] for i, off in enumerate(offsets)]):
                    continue
                in_list.append(offsets)

    n_splits = len(gpu_list)
    out_list = []
    out_list = [in_list[i::n_splits] for i in range(n_splits)]

    for ii, olist in enumerate(out_list):
        list_name = os.path.join(save_folder, './list_gpu_%i.json' % gpu_list[ii])
        with open(list_name, 'w') as f:
            json.dump(olist, f)
