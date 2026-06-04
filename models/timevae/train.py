import numpy as np
import argparse
import torch
from ..data_utils import *
from time import time
import os
from .src.vae_pipeline import train_vae_pipeline
import time

def main(args):

    data = load_data(args.dataname)
    print(data.shape)

    ckpt_dir = f"models/timevae/checkpoints/{args.dataname}"
    os.makedirs(ckpt_dir,exist_ok=True)

    epochs = 200

    start_time = time.perf_counter()
    model = train_vae_pipeline(data,'timevae',ckpt_path=ckpt_dir,vae_epochs=epochs)
    end_time = time.perf_counter()

    runtime_seconds = end_time - start_time
    

    runtime_path = f"{ckpt_dir}/runtime.txt"
    with open(runtime_path, "w") as f:
        f.write(f"runtime_seconds: {runtime_seconds:.4f}\n")
        f.write(f"runtime_minutes: {runtime_seconds / 60:.4f}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"dataset: {args.dataname}\n")
        f.write(f"model: TimeVAE\n")
    print(f"Saved runtime to {runtime_path}")
    #print(end-start)
    ###################


    #print(args.save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TimeVAE')

    parser.add_argument('--dataname', type=str, default='sine', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'