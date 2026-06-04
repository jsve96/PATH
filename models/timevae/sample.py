import numpy as np
import argparse
import torch
from ..data_utils import *
from time import time
import os
from .src.vae_pipeline import sample_vae_pipeline
import time

def main(args):
    
    data = load_data(args.dataname)

    test_data = load_data(args.dataname, train=False)
    M,N,d = test_data.shape
    print(data.shape)

    ckpt_dir = f"models/timevae/checkpoints/{args.dataname}"
    os.makedirs(ckpt_dir,exist_ok=True)

    generated_data = sample_vae_pipeline(data,M,'timevae',ckpt_path=ckpt_dir)

    np.save(args.save_path, generated_data)
    print(f'Generated {M} samples -> {args.save_path}.npy')


    

    

   
    #print(end-start)
    ###################


    #print(args.save_path)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sampling of TimeVAE')

    parser.add_argument('--dataname', type=str, default='sine', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'