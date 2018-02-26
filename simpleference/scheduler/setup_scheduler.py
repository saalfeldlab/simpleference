import click
import time
import os
import json
from collections import deque
from multiprocessing import Lock
import threading

from distributed.diagnostics.plugin import SchedulerPlugin


# This is a naive implementation of the block service to provide multiple (distributed)
# gpu workers with block coordinates for inference.
# I don't actually know if this does work as intended (i.e. if the clients can indeed
# call `request_block` and `confirm_block` on the scheduler).
#
# Also, it would be more elegant to implement this using the dask.distributed submit
# mechanism, to handle errors via the futures that are returned by this.
# But I don't understand this well enough yet.

# TODO tear down that serializes the processed and failed blocks
class BlockServicePlugin(SchedulerPlugin):
    # the block time limit TODO should be a parameter
    time_limit = 600
    # the time frame for checking blocks TODO parameter
    check_interval = 60

    def __init__(self, block_file):
        assert os.path.exists(block_file), block_file
        # load the coordinates of the blocks that will be processed
        # make a queue containing all block offsets
        with open(block_file, 'r') as f:
            self.block_queue = deque(json.load(f))
        # list to keep track of ids that are currently processed
        self.in_progress = []
        self.time_stamps = []
        # list of offsets that have been processed
        self.processed_list = []
        # list of failed blocks
        self.failed_blocks = []
        self.lock = Lock()
        SchedulerPlugin.__init__(self)

        # start the background thread that checks for failed jobs
        # TODO can this also be done with mp ?!
        bg_thread = threading.Thread(target=self.check_progress_list, args=())
        bg_thread.daemon = True
        bg_thread.start()

    # check the progress list for blokcs that have exceeded the time limit
    def check_progress_list(self):
        while True:
            time.sleep(self.check_interval)
            with self.lock:
                now = time.time()
                # find blocks that have exceeded the time limit
                failed_block_ids = [ii for ii, time_stamp in enumerate(self.time_stamps)
                                    if now - time_stamp > self.time_limit]
                # remove failed blocks and time stamps from in progress and
                # append failed blocks to the failed list
                # NOTE: we need to iterate in reverse order to delete the correct elements
                for ii in sorted(failed_block_ids, reverse=True):
                    del self.time_stamps[ii]
                    self.failed_blocks.append(self.in_progress.pop(ii))

    # request the next block to be processed
    # if no more blocks are present, return None
    def request_block(self):
        with self.lock:
            if len(self.block_queue) > 0:
                block_offsets = self.block_queue.pop()
                self.in_progress.append(block_offsets)
                self.time_stamps.append(time.time())
            else:
                block_offsets = None
                # TODO repopulate with failed blocks ?!
        return block_offsets

    # confirm that a block has been processed
    def confirm_block(self, block_offset):
        # see of the offset is still in the in-progress
        # list and remove it.
        # if not, the time limit was exceeded and something is most likely wrong
        # with the block and the block was put on the failed block list
        try:
            index = self.in_progress.index(block_offset)
            with self.lock:
                del self.in_progress[index]
                del self.time_stamps[index]
                self.processed_list.append(block_offset)
        except ValueError:
            pass


# TODO add time-limit and check interval as options
@click.command()
@click.argmument('block_file', type=click.Path(exists=True))
def dask_setup(scheduler, block_file):
    plugin = BlockServicePlugin(block_file)
    scheduler.add_plugin(plugin)
