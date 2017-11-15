import os
import numpy as np
from gunpowder.caffe.net_io_wrapper import NetIoWrapper
from gunpowder.ext import caffe


class CaffePredict(object):
    '''Augments a batch with network predictions.

    Args:

        prototxt (string): Filename of the network prototxt.

        weights (string): Filename of the network weights.

        input_key (string): Name of the input layer of the network.

        output_key (string): Name of the output layer of the network.

        gpu (int): Which GPU to use.
    '''

    def __init__(self,
                 prototxt,
                 weights,
                 input_key,
                 output_key,
                 gpu):

        # TODO validate that gpu is available
        assert os.path.exists(prototxt)
        assert os.path.exists(weights)
        for f in [prototxt, weights]:
            if not os.path.isfile(f):
                raise RuntimeError("%s does not exist" % f)
        self.prototxt = prototxt
        self.weights = weights
        self.input_key = input_key
        self.output_key = output_key

        caffe.enumerate_devices(False)
        caffe.set_devices((gpu,))
        caffe.set_mode_gpu()
        caffe.select_device(gpu, False)

        self.net = caffe.Net(self.prototxt, self.weights, caffe.TEST)
        self.net_io = NetIoWrapper(self.net, [self.output_key])

    def __call__(self, input_data):
        assert isinstance(input_data, np.ndarray)
        self.net_io.set_inputs({self.input_key: input_data})

        self.net.forward()
        output = self.net_io.get_outputs()[self.output_key]
        assert isinstance(output, np.ndarray)
        # remove batch-dimension
        if output.ndim == 5:
            output = output[0]
        assert output.ndim == 4
        return output.astype('float32')
