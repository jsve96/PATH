# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import autograd
import numpy as np
import time

from autoencoder import Autoencoder
from gan import Generator, Discriminator


def _seq_len_to_mask(seq_len: torch.Tensor, max_len: int) -> torch.Tensor:
    """Return boolean mask (N, T); True where position < length."""
    idx = torch.arange(max_len, device=seq_len.device).unsqueeze(0)
    return idx < seq_len.unsqueeze(1)


def _make_loader(data: np.ndarray, batch_size: int, shuffle: bool, drop_last: bool = True) -> DataLoader:
    t = torch.tensor(data, dtype=torch.float32)
    return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


class AeGAN:
    def __init__(self, d: int, params: dict):
        """
        Args:
            d      : feature dimension of time series (no static features)
            params : dict with keys device, logger, root_dir, hidden_dim,
                     embed_dim, layers, dropout, ae_batch_size, gan_batch_size,
                     ae_lr, gan_lr, gan_alpha, noise_dim, weight_decay
        """
        self.params = params
        self.device = params["device"]
        self.logger = params["logger"]
        self.d = d

        self.ae = Autoencoder(
            d,
            params["hidden_dim"],
            params["embed_dim"],
            params["layers"],
            dropout=params["dropout"],
        ).to(self.device)

        self.ae_optm = torch.optim.Adam(
            params=self.ae.parameters(),
            lr=params["ae_lr"],
            betas=(0.9, 0.999),
            weight_decay=params["weight_decay"],
        )

        self.loss_mse = nn.MSELoss(reduction="none")

        self.generator = Generator(
            params["noise_dim"], params["hidden_dim"], params["layers"]
        ).to(self.device)

        self.discriminator = Discriminator(params["embed_dim"]).to(self.device)

        self.discriminator_optm = torch.optim.RMSprop(
            params=self.discriminator.parameters(),
            lr=params["gan_lr"],
            alpha=params["gan_alpha"],
        )
        self.generator_optm = torch.optim.RMSprop(
            params=self.generator.parameters(),
            lr=params["gan_lr"],
            alpha=params["gan_alpha"],
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def load_ae(self, path: str = None):
        path = path or f"{self.params['root_dir']}/ae.dat"
        self.logger.info(f"load ae: {path}")
        self.ae.load_state_dict(torch.load(path, map_location=self.device))

    def load_generator(self, path: str = None):
        path = path or f"{self.params['root_dir']}/generator.dat"
        self.logger.info(f"load generator: {path}")
        self.generator.load_state_dict(torch.load(path, map_location=self.device))

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _dyn_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Masked MSE over (N, T, d) tensors."""
        loss = self.loss_mse(pred, target)                            # (N, T, d)
        mask = _seq_len_to_mask(seq_len, target.size(1)).unsqueeze(-1).float()
        return (loss * mask).sum() / (mask.sum() * self.d)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_ae(self, data: np.ndarray, epochs: int = 1000):
        """Train the autoencoder.

        Args:
            data  : (N, T, d) float32 numpy array, already normalised
            epochs: number of full passes
        """
        _, T, _ = data.shape
        loader = _make_loader(data, self.params["ae_batch_size"], shuffle=True)

        for epoch in range(epochs):
            self.ae.train()
            total_loss, n_batches = 0.0, 0
            t0 = time.time()

            for (batch_dyn,) in loader:
                batch_dyn = batch_dyn.to(self.device)
                bs = batch_dyn.size(0)
                seq_len = torch.full((bs,), T, dtype=torch.long, device=self.device)

                self.ae.zero_grad()
                _, out_dyn = self.ae(batch_dyn, seq_len)
                loss = self._dyn_loss(out_dyn, batch_dyn, seq_len)
                loss.backward()
                self.ae_optm.step()

                total_loss += loss.item()
                n_batches += 1

            if epoch % 100 == 0:
                self.logger.info(
                    f"AE Epoch {epoch + 1:>5}/{epochs}  "
                    f"loss={total_loss / n_batches:.6f}  "
                    f"t={time.time() - t0:.1f}s"
                )

        torch.save(self.ae.state_dict(), f"{self.params['root_dir']}/ae.dat")

    def train_gan(self, data: np.ndarray, iterations: int = 15000, d_update: int = 5):
        """Train the WGAN-GP generator.

        Args:
            data      : (N, T, d) float32 numpy array
            iterations: generator update steps
            d_update  : discriminator updates per generator update
        """
        _, T, _ = data.shape
        loader = _make_loader(data, self.params["gan_batch_size"], shuffle=True)

        self.discriminator.train()
        self.generator.train()
        self.ae.train()

        reg = torch.tensor(0.0)  # kept for logging on first iteration

        for iteration in range(iterations):
            t0 = time.time()
            avg_d_loss = 0.0

            # ---- discriminator updates ----
            toggle_grad(self.generator, False)
            toggle_grad(self.discriminator, True)

            for _ in range(d_update):
                for (batch_dyn,) in loader:
                    batch_dyn = batch_dyn.to(self.device)
                    bs = batch_dyn.size(0)
                    seq_len = torch.full((bs,), T, dtype=torch.long, device=self.device)

                    self.discriminator_optm.zero_grad()
                    z = torch.randn(bs, self.params["noise_dim"], device=self.device)

                    real_rep = self.ae.encoder(batch_dyn, seq_len)
                    d_real = self.discriminator(real_rep)
                    (-d_real.mean()).backward()

                    with torch.no_grad():
                        x_fake = self.generator(z)
                    x_fake.requires_grad_()
                    d_fake = self.discriminator(x_fake)
                    d_fake.mean().backward()

                    reg = 10 * self.wgan_gp_reg(real_rep, x_fake)
                    reg.backward()

                    self.discriminator_optm.step()
                    avg_d_loss += (d_fake.mean() - d_real.mean()).item()
                    break  # one batch per d_update step

            avg_d_loss /= d_update

            # ---- generator update ----
            toggle_grad(self.generator, True)
            toggle_grad(self.discriminator, False)

            self.generator_optm.zero_grad()
            z = torch.randn(self.params["gan_batch_size"], self.params["noise_dim"], device=self.device)
            g_loss = -self.discriminator(self.generator(z)).mean()
            g_loss.backward()
            self.generator_optm.step()

            if iteration % 1000 == 999:
                self.logger.info(
                    f"[{iteration + 1:>5}/{iterations}]  "
                    f"D={avg_d_loss:.4f}  G={g_loss.item():.4f}  "
                    f"GP={reg.item():.4f}  t={time.time() - t0:.2f}s"
                )

        torch.save(self.generator.state_dict(), f"{self.params['root_dir']}/generator.dat")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def synthesize(self, n: int, seq_len: int, batch_size: int = 500) -> np.ndarray:
        """Generate n synthetic samples.

        Returns:
            numpy array of shape (n, seq_len, d)
        """
        self.ae.decoder.eval()
        self.generator.eval()

        chunks = []
        remaining = n
        while remaining > 0:
            count = min(batch_size, remaining)
            with torch.no_grad():
                z = torch.randn(count, self.params["noise_dim"], device=self.device)
                hidden = self.generator(z)
                chunk = self.ae.decoder.generate_dynamics(hidden, seq_len)
            chunks.append(chunk)
            remaining -= count

        return np.concatenate(chunks, axis=0)

    def eval_ae(self, data: np.ndarray) -> tuple:
        """Reconstruct data through the autoencoder.

        Returns:
            (recon, hidden) where recon is (N, T, d) and hidden is (N, embed_dim)
        """
        _, T, _ = data.shape
        loader = _make_loader(data, self.params["gan_batch_size"], shuffle=False, drop_last=False)
        recon_list, hidden_list = [], []

        self.ae.eval()
        for (batch_dyn,) in loader:
            batch_dyn = batch_dyn.to(self.device)
            bs = batch_dyn.size(0)
            seq_len = torch.full((bs,), T, dtype=torch.long, device=self.device)
            with torch.no_grad():
                h = self.ae.encoder(batch_dyn, seq_len)
                rec = self.ae.decoder.generate_dynamics(h, T)
            hidden_list.append(h.cpu())
            recon_list.append(rec)

        return np.concatenate(recon_list, axis=0), torch.cat(hidden_list, dim=0).numpy()

    # ------------------------------------------------------------------
    # WGAN-GP regularisation
    # ------------------------------------------------------------------

    def wgan_gp_reg(
        self,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        center: float = 1.0,
    ) -> torch.Tensor:
        bs = x_real.size(0)
        eps = torch.rand(bs, 1, device=self.device)
        x_interp = ((1 - eps) * x_real + eps * x_fake).detach().requires_grad_(True)
        d_out = self.discriminator(x_interp)
        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()
        return reg


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def toggle_grad(model: nn.Module, requires_grad: bool):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
    bs = x_in.size(0)
    grad = autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad.pow(2).view(bs, -1).sum(1)
