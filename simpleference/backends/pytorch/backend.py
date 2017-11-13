import os
import numpy as np

import dill
import torch
import torch.nn as nn
from torch.autograd import Variable


class PyTorchPredict(object):
    def __init__(self, model_path, crop_prediction=None):
        assert os.path.exists(model_path)
        self.model = torch.load(model_path, pickle_module=dill)
        # NOTE we always set CUDA_VISIBLE_DEVICES to our desired gpu
        # so we can always assume gpu 0 here
        self.gpu = 0
        self.model.cuda(self.gpu)
        self.crop_prediction = crop_prediction


    def crop(self, out):
        pass


    def __call__(self, input_data):
        assert isinstance(input_data, dict)
        assert len(input_data) == 1
        data = input_data.values()[0]
        assert isinstance(data, np.ndarray)
        assert data.ndim == 3
        torch_data = Variable(torch.from_numpy(data[None, None]).cuda(self.gpu))
        out = self.model(torch_data).cpu().numpy().squeeze()
        if self.crop_prediction is not None:
            out = self.crop(out)
        return out
