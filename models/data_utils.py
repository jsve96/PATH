import pandas as pd
import torch
import numpy as np


def load_data(name,train=True):
    
    if train:
        try:
            data = np.load(f"./data/{name}/train.npy")
        except:
            data = torch.load(f"./data/{name}/train.pt").numpy()
    else:
        try:
            data = np.load(f"./data/{name}/test.npy")
        except:
            data = torch.load(f"./data/{name}/test.pt").numpy()

    return data