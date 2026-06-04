# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import random


def _seq_len_to_mask(seq_len: torch.Tensor, max_len: int) -> torch.Tensor:
    """Return boolean mask (N, T); True where position < length."""
    idx = torch.arange(max_len, device=seq_len.device).unsqueeze(0)
    return idx < seq_len.unsqueeze(1)


def mean_pooling(tensor: torch.Tensor, seq_len: torch.Tensor, dim: int = 1) -> torch.Tensor:
    mask = _seq_len_to_mask(seq_len, tensor.size(dim)).float().unsqueeze(-1)
    return torch.sum(tensor * mask, dim=dim) / seq_len.unsqueeze(-1).float()


def max_pooling(tensor: torch.Tensor, seq_len: torch.Tensor, dim: int = 1):
    max_len = tensor.size(dim)
    mask = _seq_len_to_mask(seq_len, max_len).unsqueeze(-1).expand_as(tensor).float()
    return torch.max(tensor + mask.le(0.5).float() * -1e9, dim=dim)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.rnn = nn.GRU(
            input_dim, hidden_dim, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.final = nn.LeakyReLU(0.2)

    def forward(self, dynamics: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        bs = dynamics.size(0)
        packed = nn.utils.rnn.pack_padded_sequence(
            dynamics, seq_len.cpu(), batch_first=True, enforce_sorted=False,
        )
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        h1, _ = max_pooling(out, seq_len)
        h2 = mean_pooling(out, seq_len)
        h3_last = h.view(self.layers, -1, bs, self.hidden_dim)[-1].view(bs, -1)
        glob = self.final(self.fc(torch.cat([h1, h2, h3_last], dim=-1)))

        h3_all = h.permute(1, 0, 2).contiguous().view(bs, -1)
        return torch.cat([glob, h3_all], dim=-1)  # (bs, hidden_dim * (1 + layers))


class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, dynamics_dim: int, layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dynamics_dim = dynamics_dim
        self.layers = layers
        self.rnn = nn.GRU(
            hidden_dim + dynamics_dim, hidden_dim, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dynamics_fc = nn.Linear(hidden_dim, dynamics_dim)

    def forward(
        self,
        embed: torch.Tensor,
        dynamics: torch.Tensor,
        seq_len: torch.Tensor,
        forcing: float = 0.5,
    ):
        bs, max_len, _ = dynamics.size()
        glob = embed[:, :self.hidden_dim].unsqueeze(1)
        hidden = (
            embed[:, self.hidden_dim:]
            .view(bs, self.layers, -1)
            .permute(1, 0, 2)
            .contiguous()
        )
        x = dynamics[:, 0:1, :]
        res = []
        for i in range(max_len):
            inp = torch.cat([glob, x.detach()], dim=-1)
            out, hidden = self.rnn(inp, hidden)
            out = torch.sigmoid(self.dynamics_fc(out))  # (bs, 1, d)
            if random.random() > forcing:
                x = out
            else:
                x = dynamics[:, i + 1:i + 2, :] if i + 1 < max_len else out
            res.append(out)
        return None, torch.cat(res, dim=1)

    def generate_dynamics(self, embed: torch.Tensor, max_len: int) -> "np.ndarray":
        bs = embed.size(0)
        glob = embed[:, :self.hidden_dim].unsqueeze(1)
        hidden = (
            embed[:, self.hidden_dim:]
            .view(bs, self.layers, -1)
            .permute(1, 0, 2)
            .contiguous()
        )
        x = torch.zeros(bs, 1, self.dynamics_dim, device=embed.device)
        res = []
        for _ in range(max_len):
            inp = torch.cat([glob, x], dim=-1)
            out, hidden = self.rnn(inp, hidden)
            x = torch.sigmoid(self.dynamics_fc(out))
            res.append(x)
        import numpy as np
        return torch.cat(res, dim=1).detach().cpu().numpy()


class Autoencoder(nn.Module):
    def __init__(
        self,
        dynamics_dim: int,
        hidden_dim: int,
        embed_dim: int,
        layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = Encoder(dynamics_dim, hidden_dim, embed_dim, layers, dropout)
        self.decoder = Decoder(hidden_dim, dynamics_dim, layers, dropout)

    def forward(self, dynamics: torch.Tensor, seq_len: torch.Tensor):
        hidden = self.encoder(dynamics, seq_len)
        # Shift right: prepend zeros as the decoder's "start token"
        input_x = torch.cat([
            torch.zeros_like(dynamics[:, 0:1, :]),
            dynamics[:, :-1, :],
        ], dim=1)
        return self.decoder(hidden, input_x, seq_len)
