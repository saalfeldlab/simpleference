import json
import os


class NamingScheme(object):
    def __init__(self, basename='/nrs/saalfeld/heinrichl/fafb_meta_copy2/', list_extension=''):
        self.basename = basename
        self.list_extension = list_extension
        self.full_list_name = 'list_gpu_%i'+list_extension+'.json'
        self.updated_list_name = 'list_gpu_%i_updated'+list_extension+'.json'
        self.processed_list_name = 'list_gpu_%i_processed'+list_extension+'.txt'
        self.curated_list_name = 'list_gpu_%i_processed_curated'+list_extension+'.json'

    def full_list(self, gpu):
        return os.path.join(self.basename, self.full_list_name % gpu)

    def updated_list(self, gpu):
        return os.path.join(self.basename, self.updated_list_name % gpu)

    def curated_list(self, gpu):
        return os.path.join(self.basename, self.curated_list_name % gpu)

    def processed_list(self, gpu):
        return os.path.join(self.basename, self.processed_list_name % gpu)

    def custom_list(self, gpu, custom_extension):
        return os.path.join(self.basename,
                            ('list_gpu_{0:}'+self.list_extension+'_{1:}.json').format(gpu, custom_extension))

    def curate_processed_list(self, gpu):
        with open(self.processed_list(gpu), 'r') as f:
            list_as_str = f.read()
        curated_list_as_str = list_as_str[:list_as_str.rfind(']') + 1]
        file_to_curated_list = self.curated_list(gpu)
        with open(file_to_curated_list, 'w') as f:
            f.write('[' + curated_list_as_str + ']')
        return file_to_curated_list

    def update_full_list(self, gpu):
        with open(self.curated_list(gpu), 'r') as f:
            processed_list = json.load(f)
            processed_list = {tuple(coo) for coo in processed_list}
        with open(self.full_list(gpu), 'r') as f:
            full_list = json.load(f)
            full_list = {tuple(coo) for coo in full_list}
        if processed_list == full_list:
            print("processing was complete, no updated offset list created for {}".format(self.full_list(gpu)))
            return
        assert processed_list < full_list

        updated_list = full_list - processed_list
        with open(self.updated_list(gpu), 'w') as f:
            print("save updated list {}".format(self.updated_list(gpu)))
            json.dump([list(coo) for coo in updated_list], f)


def curate_and_update_lists(gpu_list, list_extension=''):
    ns = NamingScheme(list_extension=list_extension)
    for gpu in gpu_list:
        ns.curate_processed_list(gpu)
        ns.update_full_list(gpu)


def combine_updated(gpu_list, list_extension):
    ns = NamingScheme(list_extension=list_extension)
    missing_list = []
    for gpu in gpu_list:
        if os.path.exists(ns.updated_list(gpu)):
            with open(ns.updated_list(gpu), 'r') as f:
                missing_list.extend(json.load(f))
        else:
            print(ns.updated_list(gpu)+" does not exist.")
    with open(ns.custom_list('all', 'missing'), 'w') as f:
        json.dump(missing_list, f)


if __name__ == '__main__':
    gpu_list = list(range(52))
    curate_and_update_lists(gpu_list, list_extension='_part2_missing')
    #combine_updated(gpu_list, '_part2')