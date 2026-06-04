
## Necessary Packages
import numpy as np
import os
import torch
import controldiffeq
import pathlib


PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent


def to_tensor(data):
    return torch.from_numpy(data).float()


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors

def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')