import numpy as np
import os
import sys
import torch
import torch.utils.data as Data

#from data.data_provider.data_factory import data_provider
#from data.long_range import parse_datasets

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data



def real_data_loading(data_name):
    """Load and preprocess real-world data.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    assert data_name in ['stock', 'energy','air','eye','ecg','traffic','kdd_cup','sine']

    data = np.load("./")

    return data


def gen_dataloader(args):
    
    ori_data = np.load(f"./data/{args.dataname}/train.npy")
    ori_data = torch.Tensor(np.array(ori_data))
    print(ori_data.shape)
    train_set = Data.TensorDataset(ori_data)

    if args.dataname in ['solar_weekly', 'fred_md', 'nn5_daily', 'temperature_rain', 'traffic_hourly', 'kdd_cup']:
        
        args.seq_len = ori_data.shape[1]     # update seq_len to match the dataset
        full_len = ori_data.shape[0]
        randperm = torch.randperm(full_len)
        train_data = ori_data[randperm[:int(full_len * 0.8)]]
        test_data = ori_data[randperm[int(full_len * 0.8):]]
        train_set = Data.TensorDataset(train_data)
        test_set = Data.TensorDataset(test_data)
        train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
        test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
        return train_loader, test_loader


    train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)

    return train_loader, train_loader


def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def stft_transform(data, args):
    import torchaudio.transforms as transforms
    data = torch.permute(data, (0, 2, 1))  # we permute to match requirements of torchaudio.transforms.Spectrogram
    n_fft = args.n_fft
    hop_length = args.hop_length
    spec = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, center=True, power=None)
    transformed_data = spec(data)
    real, min_real, max_real = MinMaxScaler(transformed_data.real.numpy(), True)
    real = (real - 0.5) * 2
    imag, min_imag, max_imag = MinMaxScaler(transformed_data.imag.numpy(), True)
    imag = (imag - 0.5) * 2
    # saving min and max values, we will need them for inverse transform
    args.min_real, args.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
    args.min_imag, args.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)
    return torch.Tensor(real), torch.tensor(imag)


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


