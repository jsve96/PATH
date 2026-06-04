import numpy as np
import argparse
import torch
from ..data_utils import *
from .model import *
from time import time
import os


def main(args):
    print(args.kernel)
    data = load_data(args.dataname)
    test_data = load_data(args.dataname,train=False)


    M,N,d = data.shape
    M_simu = test_data.shape[0] #


    #args.q = 0.001 

    if args.q <= 1:
        q_count = max(1, int(np.ceil(args.q * M)))
    else:
        q_count = int(args.q)
    print(q_count)
    q_count = min(q_count, M)



    start = time()
    syn_samples,_,_,bandwith = simulate_path(N=N,M=M,d=d,K=args.K, X=data,q=q_count,M_simu=M_simu, kernel_type=args.kernel)
    end =time()
    print(bandwith)
    print(end-start)

    print(data.shape)
    print(syn_samples.shape)

    #print(args.save_path)

    os.makedirs(f"synthetic/{args.dataname}",exist_ok=True)
    print(f'Samples save to {args.save_path}')
    np.save(args.save_path,syn_samples)
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