import numpy as np
import argparse
import torch
from ..data_utils import *
from time import time
import os
from .lib.timegan import TimeGAN
import time

def main(args):
    print(args.kernel)
    data = load_data(args.dataname)

    print(data.shape)


    M,N,d = data.shape

    M_simu = M
    args.z_dim = d
    #start = time()

    ## checkpoint dir
    args.outf = f"models/timegan/checkpoints"
    #os.makedirs(args.outf,exist_ok=True)
    args.name = args.dataname
    args.manualseed = -1
    args.isTrain = True
    start_time = time.perf_counter()
    model = TimeGAN(args,data)
    model.train()
    end_time = time.perf_counter()

    runtime_seconds = end_time - start_time
    ckpt_dir = f"models/timegan/checkpoints/{args.dataname}"

    runtime_path = f"{ckpt_dir}/runtime.txt"
    epochs = args.iteration
    with open(runtime_path, "w") as f:
        f.write(f"runtime_seconds: {runtime_seconds:.4f}\n")
        f.write(f"runtime_minutes: {runtime_seconds / 60:.4f}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"dataset: {args.dataname}\n")
        f.write(f"model: TimeGAN\n")
    print(f"Saved runtime to {runtime_path}")
    #print(end-start)
    ###################


    #print(args.save_path)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TimeGAN')

    parser.add_argument('--dataname', type=str, default='sine', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'