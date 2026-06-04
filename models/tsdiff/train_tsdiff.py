import warnings
import argparse
import numpy as np
from pathlib import Path
import time
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .data import DataModule
from .diffusion_model import DiffusionModule
from .ode_model import ODEModule
from .nf_model import NFModule
from .sde_model import SDEModule


def train(
    *,
    seed: int,
    dataset: str,
    diffusion: str,
    model: str,
    gp_sigma: float = None,
    ou_theta: float = None,
    beta_start: float = None,
    beta_end: float = None,
    batch_size: int = 256,
    hidden_dim: int = 128,
    predict_gaussian_noise: bool = True,
    beta_fn: str = 'linear',
    discrete_num_steps: int = 1000,
    continuous_t1: float = 1,
    loss_weighting: str = 'exponential',
    learning_rate: float = 1e-5,
    weight_decay: float = 0,
    epochs: int = 100,
    patience: int = 20,
    return_model: bool = False,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    datamodule = DataModule(dataset, batch_size=batch_size)

    if diffusion is not None:
        if 'Continuous' in diffusion:
            beta_start, beta_end = 0.1, 20
        else:
            beta_start, beta_end = 1e-4, 20 / discrete_num_steps

    if model == 'ode':
        Module = ODEModule
    elif model == 'nf':
        Module = NFModule
    elif model == 'sde':
        Module = SDEModule
    else:
        Module = DiffusionModule

    # Load model
    module = Module(
        dim=datamodule.dim,
        data_mean=datamodule.x_mean,
        data_std=datamodule.x_std,
        max_t=datamodule.t_max,
        diffusion=diffusion,
        model=model,
        predict_gaussian_noise=predict_gaussian_noise,
        gp_sigma=gp_sigma,
        ou_theta=ou_theta,
        beta_fn=beta_fn,
        discrete_num_steps=discrete_num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        continuous_t1=continuous_t1,
        loss_weighting=loss_weighting,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # Train

    ckpt_dir = f"models/tsdiff/checkpoints/{dataset}/{diffusion}/{model}"
    checkpointing = ModelCheckpoint(dirpath=ckpt_dir, monitor='val_loss', mode='min', save_top_k=1, filename='best-checkpoint')
    #early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=epochs,
        log_every_n_steps=1,
        enable_checkpointing=True,
        inference_mode=False,
        #callbacks=[early_stopping, checkpointing],
        callbacks=[checkpointing],
    )

    start_time = time.perf_counter()

    trainer.fit(module, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())

    end_time = time.perf_counter()

    runtime_seconds = end_time - start_time
    runtime_path = f"{ckpt_dir}/runtime.txt"

    with open(runtime_path, "w") as f:
        f.write(f"runtime_seconds: {runtime_seconds:.4f}\n")
        f.write(f"runtime_minutes: {runtime_seconds / 60:.4f}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"dataset: {dataset}\n")
        f.write(f"diffusion: {diffusion}\n")
        f.write(f"model: {model}\n")
        f.write(f"seed: {seed}\n")
    print(f"Saved runtime to {runtime_path}")


    # Load best model
    module = Module.load_from_checkpoint(checkpointing.best_model_path)

    # Evaluation
    metrics = trainer.test(module, datamodule.test_dataloader())

    # # Generate samples
    # if seed == 1:
    #     t = datamodule.trainset[:1000][1].to(device)
    #     samples = module.sample(t=t, use_ode=True)
    #     np.save(SAMPLE_DIR / f'{dataset}-{diffusion}-{model}-{gp_sigma or ou_theta}-{predict_gaussian_noise}', samples.detach().cpu().numpy())

    if return_model:
        return module, datamodule, trainer, metrics

    return metrics[0]


