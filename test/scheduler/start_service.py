import sys
import json
sys.path.append('/home/papec/Work/my_projects/z5/bld/python')
sys.path.append('/home/papec/Work/my_projects/nnets/simpleference')


# make a dummy block list
# and output file
def setup():
    import z5py
    shape = (1000, 1000, 1000)
    chunks = (100, 100, 100)
    f = z5py.File('./output.n5')
    f.create_dataset('out', shape=shape, chunks=chunks, dtype='uint8', compression='gzip')

    block_list = []
    for z in range(10):
        for y in range(10):
            for x in range(10):
                block_list.append([z*100, y*100, x*100])

    with open('block_list.json', 'w') as f:
        json.dump(block_list, f)


def start_service():
    from simpleference.scheduler import start_block_service
    host, port = "localhost", 9999
    start_block_service(host, port, './block_list.json')


if __name__ == '__main__':
    # setup()
    start_service()
