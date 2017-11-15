import os
import numpy as np

import dill
import torch
from torch.autograd import Variable


class PyTorchPredict(object):
    def __init__(self, model_path, crop_prediction=None):
        assert os.path.exists(model_path)
        self.model = torch.load(model_path, pickle_module=dill)
        # NOTE we always set CUDA_VISIBLE_DEVICES to our desired gpu
        # so we can always assume gpu 0 here
        self.gpu = 0
        self.model.cuda(self.gpu)
        # validate cropping
        if crop_prediction is not None:
            assert isinstance(crop_prediction, (list, tuple))
            assert len(crop_prediction) == 3
        self.crop_prediction = crop_prediction

    def crop(self, out):
        shape_diff = tuple((shape - crop) // 2
                           for shape, crop in zip(out.shape, self.crop_prediction))
        bb = tuple(slice(diff, shape - diff) for diff, shape in zip(shape_diff, out.shape))
        return out[bb]

    def __call__(self, input_data):
        assert isinstance(input_data, np.ndarray)
        assert input_data.ndim == 3
        torch_data = Variable(torch.from_numpy(input_data[None, None]).cuda(self.gpu))
        out = self.model(torch_data).cpu().numpy().squeeze()
        if self.crop_prediction is not None:
            out = self.crop(out)
        return out
