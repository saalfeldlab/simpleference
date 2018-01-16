import numpy as np


def normalize(data, eps=1e-4, mean=None, std=None):
    mean = data.mean() if mean is None else mean
    std = data.std() if std is None else std
    return (data - mean) / (std + eps)


def cast(data, dtype='float32'):
    return data.astype(dtype, copy=False)


def preprocess(data):
    return cast(normalize(data))
