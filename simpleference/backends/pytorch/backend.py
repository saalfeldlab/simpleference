import os
import numpy as np

import dill
import torch
from torch.autograd import Variable
import threading


class PyTorchPredict(object):
    def __init__(self, model_path, crop=None, gpu=0):
        assert os.path.exists(model_path), model_path
        self.model = torch.load(model_path, pickle_module=dill)
        # NOTE we always set CUDA_VISIBLE_DEVICES to our desired gpu
        # so we can always assume gpu 0 here
        self.gpu = gpu
        self.model.cuda(self.gpu)
        # validate cropping
        if crop is not None:
            assert isinstance(crop, (list, tuple))
            assert len(crop) == 3
        self.crop = crop
        self.lock = threading.Lock()

    def apply_crop(self, out):
        shape_diff = tuple((shape - crop) // 2
                           for shape, crop in zip(out.shape, self.crop))
        bb = tuple(slice(diff, shape - diff) for diff, shape in zip(shape_diff, out.shape))
        return out[bb]

    def __call__(self, input_data):
        assert isinstance(input_data, np.ndarray)
        assert input_data.ndim == 3
        with self.lock:
            torch_data = Variable(torch.from_numpy(input_data[None, None]).cuda(self.gpu),
                                  volatile=True)
            print('predicting a block!')
            out = self.model(torch_data).cpu().data.numpy().squeeze()
        if self.crop is not None:
            out = self.apply_crop(out)
        return out
