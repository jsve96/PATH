import numpy as np
import argparse
import torch
from ..data_utils import *
from .model import *
import time
import os


def main(args):
    print(args.kernel)
    data = load_data(args.dataname)
    test_data = load_data(args.dataname, train=False)


    M,N,d = data.shape
    data_train = np.zeros((M,N+1,d))
    data_train[:,1:] = data

    M_simu = test_data.shape[0]
    start_time = time.perf_counter()
    syn_samples = simulateSB_multi_mark(N=N,M=M,d=d,K=args.K, X=data_train,M_simu=M_simu,N_pi=100,deltati=1/252,h=args.h)
    end_time = time.perf_counter()

    runtime_seconds = end_time - start_time
    
    print(data.shape)
    print(syn_samples.shape)

    #print(args.save_path)

    os.makedirs(f"synthetic/{args.dataname}",exist_ok=True)
    print(f'Samples save to {args.save_path}')
    np.save(args.save_path,syn_samples)

    ckpt_dir = f"models/sbts/checkpoints/{args.dataname}"
    os.makedirs(ckpt_dir,exist_ok=True)

    runtime_path = f"{ckpt_dir}/runtime.txt"
    #epochs = args.iterations
    with open(runtime_path, "w") as f:
        f.write(f"runtime_seconds: {runtime_seconds:.4f}\n")
        f.write(f"runtime_minutes: {runtime_seconds / 60:.4f}\n")
        #f.write(f"epochs: {epochs}\n")
        f.write(f"dataset: {args.dataname}\n")
        f.write(f"model: SBTS\n")
    print(f"Saved runtime to {runtime_path}")

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'