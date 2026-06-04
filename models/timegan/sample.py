import numpy as np
import torch
import os
from ..data_utils import load_data
from .lib.timegan import TimeGAN


def main(args):
    data = load_data(args.dataname)
    M, N, d = data.shape
    test_data = load_data(args.dataname, train=False)

    args.outf = "models/timegan/checkpoints"
    args.name = args.dataname
    args.isTrain = False
    args.manualseed = -1
    args.resume = os.path.join(args.outf, args.dataname, 'train', 'weights')
    args.z_dim = d

    model = TimeGAN(args, data)
    model.nete.eval()
    model.netr.eval()
    model.netg.eval()
    model.netd.eval()
    model.nets.eval()

    generated = []
    M = test_data.shape[0]
    remaining = M
    with torch.no_grad():
        while remaining > 0:
            batch = min(args.batch_size, remaining)
            batch_samples = model.generation(batch)
            generated.extend(batch_samples)
            remaining -= batch

    generated_data = np.array(generated)  # [M, N, d]
    #os.makedirs(args.save_path, exist_ok=True)
    #save_file = os.path.join(args.save_path, f'{args.dataname}.npy')
    np.save(args.save_path, generated_data)
    print(f'Generated {M} samples -> {args.save_path}.npy')
