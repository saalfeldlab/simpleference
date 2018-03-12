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
        self.gpu = gpu
        self.model.cuda(self.gpu)
        # validate cropping
        if crop is not None:
            assert isinstance(crop, (list, tuple))
            assert len(crop) == 3
        self.crop = crop
        self.lock = threading.Lock()

    def apply_crop(self, out):
        shape = out.shape if out.ndim == 3 else out.shape[1:]
        shape_diff = tuple((sha - crop) // 2
                           for sha, crop in zip(shape, self.crop))
        bb = tuple(slice(diff, shape - diff) for diff, shape in zip(shape_diff, shape))
        if out.ndim == 4:
            bb = (slice(None),) + bb
        return out[bb]

    def __call__(self, input_data):
        assert isinstance(input_data, np.ndarray)
        assert input_data.ndim == 3
        # Note: in the code that follows, the GPU is locked for the 3 steps:
        # CPU -> GPU, GPU inference, GPU -> CPU. It may well be that we get
        # better performance by only locking in step 2, or steps 1-2, or steps
        # 2-3. We should perform this experiment and then choose the best
        # option for our hardware (and then remove this comment! ;)
        with self.lock:
            # 1. Transfer the data to the GPU
            torch_data = Variable(torch.from_numpy(input_data[None, None])
                                  .cuda(self.gpu), volatile=True)
            print('predicting a block!')
            # 2. Run the model
            predicted_on_gpu = self.model(torch_data)
            if isinstance(predicted_on_gpu, tuple):
                predicted_on_gpu = predicted_on_gpu[0]
            # 3. Transfer the results to the CPU
            out = predicted_on_gpu.cpu().data.numpy().squeeze()
        if self.crop is not None:
            out = self.apply_crop(out)
        return out


class InfernoPredict(PyTorchPredict):
    def __init__(self, model_path, crop=None, gpu=0, use_best=True):
        from inferno.trainers.basic import Trainer
        assert os.path.exists(model_path), model_path
        trainer = Trainer().load(from_directory=model_path, best=use_best)
        self.model = trainer.model.cuda(gpu)
        self.gpu = gpu
        # validate cropping
        if crop is not None:
            assert isinstance(crop, (list, tuple))
            assert len(crop) == 3
        self.crop = crop
        self.lock = threading.Lock()
