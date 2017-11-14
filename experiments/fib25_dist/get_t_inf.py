import os
import numpy as np


def get_t_inf(folder):
    files = os.listdir(folder)
    t_infs = []
    for ff in files:
        if not ff.startswith('t-inf'):
            continue
        with open(os.path.join(folder, ff), 'r') as f:
            line = f.readline()
        t_infs.append(float(line.split()[-2]))
    return np.max(t_infs), np.mean(t_infs), np.min(t_infs), np.std(t_infs)


if __name__ == '__main__':
    sample = 'C+'
    folder = '/nrs/saalfeld/heinrichl/segmentation/distance_thirdtest/fib25_sub_prediction_at_400000/'
    print(get_t_inf(folder))
