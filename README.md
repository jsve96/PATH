# Time Series Synthesis Benchmark

Benchmarking framework for multivariate time series generation. Supports SOTA methods evaluated against real data using discriminative and predictive metrics.

## Project Structure

```
├── main.py                  # Entry point for training and sampling
├── run_eval_disc.py         # Discriminative evaluation
├── run_eval_pred.py         # Predictive evaluation
├── run_eval_privacy.py      # Privacy evaluation
├── data/                    # Real datasets (.npy)
├── synthetic/               # Generated samples (.npy)
├── results/                 # Evaluation results (.csv)
│   ├── disc/
│   └── pred/
|   └── privacy/
└── models/
    ├── path/                # PATH (non-parametric, kernel-based)
    ├── sbts/                # SBTS (Schrödinger Bridge)
    ├── timegan/             # TimeGAN (GRU/LSTM-based GAN)
    ├── tsdiff/              # DSPD-GP, DSPD-OU, CSPD-GP, CSPD-OU
    ├── timevae/             # TimeVAE  
    ├── rtsgan/              # RTSGAN 
    ├── imagentime/          # Imagentime                        
    └── kovae/               # KoVAE 
```

## Datasets

Place datasets (csv, txt files) in  `data/`

Available: `sine`, `stocks`, `energy`, `air`, `ecg`, `eye`, `kdd_cup`, `traffic`

stocks, energy from [Repo](https://github.com/jsyoon0823/TimeGAN/tree/master/data)


Run 

```bash
python utils/generate.py
```
to generate train/test splits which will be placed into `data/{dataset}/train.npy test.npy`

## Generation

### PATH and SBTS

Non-parametric methods, no separate train/sample step:

```bash
python main.py --dataname stocks --method path
python main.py --dataname stocks --method sbts
```

generates samples which will be placed into `synthetic\dataname\method.npy`

### Other methods

```bash
# Train
python main.py --dataname stocks --method name --mode train

# Sample
python main.py --dataname stocks --method name --mode sample
```

Checkpoints saved to `models/name/checkpoints/{dataname}`.

### TSDiff

Score-based diffusion model. Requires selecting a diffusion process and backbone architecture.

```bash
# Train
python main.py --dataname stocks --method tsdiff \
  --diffusion GaussianDiffusion --model rnn \
  --epochs 1000 --patience 1000 --discrete_num_steps 1000

# Sample
python main.py --dataname stocks --method tsdiff \
  --diffusion GaussianDiffusion --model rnn --mode sample
```

Checkpoints saved to `models/tsdiff/checkpoints/{dataname}/{diffusion}/{model}/best-checkpoint.ckpt`.

Output is saved to `synthetic/{dataname}/{method}.npy`.

## Key Arguments

### General

| Argument | Default | Description |
|---|---|---|
| `--dataname` | `sine` | Dataset name |
| `--method` | — | `path`, `sbts`, `timegan`, `tsdiff` |
| `--mode` | `train` | `train` or `sample` |
| `--gpu` | `0` | GPU index (`-1` for CPU) |


## Evaluation

All scripts read from `data/{dataname}.npy` and `synthetic/{dataname}/{method}.npy` and write summary statistics to `results/{disc,pred,privacy}/{dataname}/{method}.csv`.

### Discriminative score

Trains GRU/TCN and LR/RF classifiers to distinguish real from synthetic data. Lower accuracy → better synthesis.

```bash
python run_eval_disc.py --dataname stocks --method name
```

### Predictive score

Evaluates TSTR/TRTR multi-horizon predictive error, ACF error, and cross-correlation error. Both TSTR and TRTR are compared against a persistence baseline (predict last observed value at every horizon); skill scores reveal whether the predictor adds value beyond persistence.

```bash
python run_eval_pred.py --dataname stocks --method name

# Use MSE/RMSE to penalise large per-feature errors more strongly
python run_eval_pred.py --dataname stocks --method tsdiff --metric rmse
```

### Privacy score

Measures memorization along three complementary axes:

1. **Nearest-Neighbour Distance (NND)** — compares `min d(synthetic, train)` against `min d(real-test, train)`. If synthetic trajectories cluster closer to training data than held-out real trajectories, this signals memorisation.
2. **Membership Inference Attack (MIA)** — an attacker uses each real trajectory's distance to the nearest synthetic sample as a feature to decide whether it came from the training set. Reports ROC-AUC over the member/non-member classification task.
3. **Subsequence Copying** — for each generated trajectory, finds the longest contiguous block that near-matches any training trajectory across all temporal offsets. High `lmax_norm_mean` indicates direct sequence copying.

```bash
# Full evaluation (NND + MIA + subsequence)
python run_eval_privacy.py --dataname stocks --method name

# NND + MIA only (skip slow subsequence check)
python run_eval_privacy.py --dataname stocks --method name --no-subseq
```

### Key arguments (all eval scripts)

| Argument | Default | Description |
|---|---|---|
| `--seeds` | `42,0,1,2,3` | Comma-separated seeds for multiple runs |
| `--test-ratio` | `0.2` | Held-out test split |
| `--real-start` | `0` | Time-axis start index for real data |
| `--syn-start` | `0` | Time-axis start index for synthetic data |


## Add a new method

If you want to add your own time series generation method create a new directory in `models/` and create a `train.py` and `sample.py` file. 

Make sure you wrap your method such that npy files (N,T,d) are accepted. 

```python
import numpy as np
import argparse
from ..data_utils import *
from time import time
import os
from .your_model import train
import time

def main(args):

    data = load_data(args.dataname)
    print(data.shape)

    ckpt_dir = f"models/your_model/checkpoints/{args.dataname}"
    os.makedirs(ckpt_dir,exist_ok=True)

    start_time = time.perf_counter()
    model = train(data,ckpt_path=ckpt_dir)
    end_time = time.perf_counter()

    runtime_seconds = end_time - start_time
    

    runtime_path = f"{ckpt_dir}/runtime.txt"
    with open(runtime_path, "w") as f:
        f.write(f"runtime_seconds: {runtime_seconds:.4f}\n")
        f.write(f"runtime_minutes: {runtime_seconds / 60:.4f}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"dataset: {args.dataname}\n")
        f.write(f"model: your_model\n")
    print(f"Saved runtime to {runtime_path}")
    #print(end-start)
    ###################


    #print(args.save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of your model')

    parser.add_argument('--dataname', type=str, default='sine', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
```

The sample file should be similar to

```python
import numpy as np
import argparse
import torch
from ..data_utils import *
import os
from .your_model import sample_pipeline

def main(args):
    
    data = load_data(args.dataname)

    test_data = load_data(args.dataname, train=False)
    M,N,d = test_data.shape
    print(data.shape)

    ckpt_dir = f"models/your_model/checkpoints/{args.dataname}"
    os.makedirs(ckpt_dir,exist_ok=True)

    generated_data = sample_pipeline(data,M,ckpt_path=ckpt_dir)

    np.save(args.save_path, generated_data)
    print(f'Generated {M} samples -> {args.save_path}.npy')


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
```