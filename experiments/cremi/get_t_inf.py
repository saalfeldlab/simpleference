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
    return np.max(t_infs)


if __name__ == '__main__':
    sample = 'C+'
    folder = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/prediction_blocks_%s' % sample
    print(get_t_inf(folder))
