import warnings
import argparse
import numpy as np
from pathlib import Path
import torch
from . train_tsdiff import *


warnings.simplefilter(action='ignore', category=(np.VisibleDeprecationWarning))
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#SAMPLE_DIR = f'data/samples'
#SAMPLE_DIR.mkdir(exist_ok=True, parents=True)



def main(args):

    args.seed = 1
    print(args.seed)
    
    results = train(seed=args.seed, dataset=args.dataname, diffusion=args.diffusion, model=args.model,
                    gp_sigma=args.gp_sigma, ou_theta=args.ou_theta, epochs=args.epochs, patience=args.patience,
                    discrete_num_steps=args.discrete_num_steps)



if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Train forecasting model.')
    parser = argparse.ArgumentParser(description='Training of TSDIFF')

    parser.add_argument('--dataname', type=str, default='sine', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    args = parser.parse_args()


