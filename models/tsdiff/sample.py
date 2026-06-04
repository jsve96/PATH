import numpy as np
import torch
import os
from ..data_utils import load_data
from .diffusion_model import DiffusionModule
from .ode_model import ODEModule
from .nf_model import NFModule
from .sde_model import SDEModule


def reverse_normalize(data, min, max) -> np.ndarray:
    return (data * (max - min) + min)          # to turn back into real scale


def main(args):
    data = load_data(args.dataname)
    M, N, d = data.shape
    M,_,_ = load_data(args.dataname, train=False).shape



    device = torch.device(args.device)

    # Time tensor matches what DataModule builds during training
    t = torch.arange(N).float().view(1, N, 1).repeat(M, 1, 1).to(device)

    if args.model == 'ode':
        Module = ODEModule
    elif args.model == 'nf':
        Module = NFModule
    elif args.model == 'sde':
        Module = SDEModule
    else:
        Module = DiffusionModule

    ckpt_path = (
        f"models/tsdiff/checkpoints/{args.dataname}"
        f"/{args.diffusion}/{args.model}/best-checkpoint.ckpt"
    )
    print(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    module = Module.load_from_checkpoint(ckpt_path, map_location=device)
    module.to(device)
    module.eval()

    with torch.no_grad():
        samples = module.sample(t, use_ode= True)  # (M, N, d)
        #samples = reverse_normalize(samples,min=data.min(),max=data.max())
    

    save_path = f'synthetic/{args.dataname}/{args.diffusion}_{args.model}'
    os.makedirs(f"synthetic/{args.dataname}",exist_ok=True)
    np.save(save_path, samples.cpu().numpy())
    print(f'Generated {M} samples -> {save_path}.npy')
