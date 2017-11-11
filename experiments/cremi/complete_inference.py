from subprocess import call
from simpleference.inference.util import get_offset_lists


def complete_inference(sample, gpu_list, iteration):
    save_folder = './offsets_sample%s' % sample
    # TODO padded realigned volumes
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/mala_jan_original/raw/sample_%s.h5' % sample
    get_offset_lists(raw_path, gpu_list, save_folder)
    for gpu in gpu_list:
        call(['./run_inference.sh', sample, str(gpu), str(iteration)])


if __name__ == '__main__':
    sample = 'A+'
    gpu_list = range(8)
    iteration = 100000
    complete_inference(sample, gpu_list, iteration)
