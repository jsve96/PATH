from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
import warnings
import math
from torchtyping import TensorType
from typing import List, Callable


def get_beta_scheduler(name: str) -> Callable:
    if name == 'linear':
        return BetaLinear

def get_loss_weighting(name: str) -> Callable:
    if name == 'exponential':
        return exponential_loss_weighting

class BetaLinear(nn.Module):
    """
    Linear scheduling for beta.
    Input t is always from interval [0, 1].

    Args:
        start: Lower bound (float)
        end: Upper bound (float)
    """
    def __init__(self, start: float, end: float):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, t: Tensor) -> Tensor:
        return self.start * (1 - t) + self.end * t

    def integral(self, t: Tensor) -> Tensor:
        return 0.5 * (self.end - self.start) * t.square() + self.start * t


def exponential_loss_weighting(beta_fn, i):
    return 1 - torch.exp(-beta_fn.integral(i))


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_value: float):
        super().__init__()
        self.max_value = max_value

        linear_dim = dim // 2
        periodic_dim = dim - linear_dim

        self.scale = torch.exp(-2 * torch.arange(0, periodic_dim).float() * math.log(self.max_value) / periodic_dim)
        self.shift = torch.zeros(periodic_dim)
        self.shift[::2] = 0.5 * math.pi

        self.linear_proj = nn.Linear(1, linear_dim)

    def forward(self, t: TensorType[..., 'length', 1]) -> TensorType[..., 'length', 'dim']:
        periodic = torch.sin(t * self.scale.to(t) + self.shift.to(t))
        linear = self.linear_proj(t / self.max_value)

        return torch.cat([linear, periodic], -1)
    

class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: Callable = nn.ReLU(),
        final_activation: Callable = None,
    ):
        super().__init__()

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)

        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(activation)
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: TensorType[..., 'in_dim']) -> TensorType[..., 'out_dim']:
        return self.net(x)
    


from typing import Any, Callable, Tuple, Optional, Union
from torchtyping import TensorType
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td

from torchsde import sdeint
from torchdiffeq import odeint


from typing import Union
from torchtyping import TensorType

import numpy as np
import scipy.fftpack
from functools import lru_cache

import torch
import torch.nn as nn


class Normal(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim

    def forward(self, *shape, **kwargs):
        return torch.randn(*shape, self.dim)

    def covariance(self, **kwargs):
        return torch.eye(self.dim)


class Wiener(nn.Module):
    """
    Wiener process / Brownian motion.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        t: Union[TensorType['seq_len'], TensorType[..., 'seq_len', 1]],
        **kwargs,
    ) -> Union[TensorType['seq_len'], TensorType[..., 'seq_len', 'dim']]:
        one_dimensional = len(t.shape) == 1

        if one_dimensional:
            t = t.unsqueeze(-1)
        t = t.repeat_interleave(self.dim, dim=-1)

        dt = torch.diff(t, dim=-2, prepend=torch.zeros_like(t[...,:1,:]).to(t))
        dw = torch.randn_like(dt) * dt.clamp(1e-5).sqrt()
        w = dw.cumsum(dim=-2)

        if one_dimensional and self.dim == 1:
            w = w.squeeze(-1)
        return w


class OrnsteinUhlenbeck(nn.Module):
    """
    Ornstein-Uhlenbeck process.

    Args:
        theta: Diffusion param, higher value = spikier (float)
    """
    def __init__(self, dim: int, theta: float = 0.5):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.wiener = Wiener(dim)

    def forward(
        self,
        *args,
        t: TensorType[..., 'seq_len', 1],
        **kwargs,
    ) -> TensorType[..., 'seq_len', 'dim']:

        delta = torch.diff(t, dim=-2, prepend=torch.zeros_like(t[...,:1,:]))
        coeff = torch.exp(-self.theta * delta)

        sample = []

        x = torch.randn(*t.shape[:-2], 1, self.dim).to(t)
        for i in range(coeff.shape[-2]):
            z = torch.randn(*t.shape[:-2], 1, self.dim).to(t)
            c = coeff[...,i,None,:]
            x = c * x + torch.sqrt(1 - c**2) * z
            sample.append(x)

        sample = torch.cat(sample, dim=-2)
        return sample

    def covariance(
        self,
        t: TensorType[..., 'seq_len', 1],
        diag_epsilon: float = 1e-4,
        **kwargs,
    ) -> TensorType[..., 'seq_len', 'seq_len']:
        t = t.squeeze(-1)
        diag = torch.eye(t.shape[-1]).to(t) * diag_epsilon
        cov = torch.exp(-(t.unsqueeze(-1) - t.unsqueeze(-2)).abs() * self.theta)
        return cov + diag

    def covariance_cholesky(self, t: TensorType[..., 'seq_len', 1]) -> TensorType[..., 'seq_len', 'seq_len']:
        return torch.linalg.cholesky(self.covariance(t))

    def covariance_inverse(self, t: TensorType[..., 'seq_len', 1]) -> TensorType[..., 'seq_len', 'seq_len']:
        return torch.linalg.inv(self.covariance(t))


class GaussianProcess(nn.Module):
    """
    Gaussian random field for one-dimensional (temporal) data.
    """
    def __init__(self, dim: int, sigma: float = 0.1):
        super().__init__()
        self.dim = dim
        self.sigma = sigma

    def forward(
        self,
        *args,
        t: TensorType[..., 'N', 1],
        **kwargs,
    ) -> TensorType[..., 'N', 'dim']:
        # If N is very large this could become slow
        # In that case, consider using sparse GP
        L = self.covariance_cholesky(t)
        e = torch.randn(*t.shape[:-1], self.dim).to(t)
        return L @ e

    def covariance(
        self,
        t: TensorType[..., 'N', 1],
        diag_epsilon: float = 1e-4,
        **kwargs,
    ) -> TensorType[..., 'N', 'N']:
        if t.shape[-1] != 1 or len(t.shape) < 2:
            t = t.unsqueeze(-1)
        distance = t - t.transpose(-1, -2)
        diag = torch.eye(t.shape[-2]).to(t) * diag_epsilon
        return torch.exp(-torch.square(distance / self.sigma)) + diag

    def covariance_cholesky(self, t: TensorType[..., 'N', 1]) -> TensorType[..., 'N', 'N']:
        return torch.linalg.cholesky(self.covariance(t))

    def covariance_inverse(self, t: TensorType[..., 'N', 1]) -> TensorType[..., 'N', 'N']:
        return torch.linalg.inv(self.covariance(t))



class ContinuousDiffusion(nn.Module):
    """
    Continuous diffusion using SDEs (https://arxiv.org/abs/2011.13456)

    Args:
        dim: Dimension of data
        beta_fn: Scheduler for noise levels
        t1: Final diffusion time
        noise_fn: Type of noise
        predict_gaussian_noise: Whether to approximate score with unit normal
        loss_weighting: Function returning loss weights given diffusion time
    """
    def __init__(
        self,
        dim: int,
        beta_fn: Callable,
        t1: float = 1.0,
        noise_fn: Callable = None,
        loss_weighting: Callable = None,
        is_time_series: bool = False,
        predict_gaussian_noise: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.t1 = t1
        self.predict_gaussian_noise = predict_gaussian_noise
        self.is_time_series = is_time_series

        self.beta_fn = beta_fn
        self.noise = noise_fn
        self.loss_weighting = partial(loss_weighting or (lambda beta, i: 1), beta_fn)

    def forward(
        self,
        x: TensorType[..., 'dim'],
        i: TensorType[..., 1],
        _return_all: Optional[bool] = False, # For internal use only
        **kwargs,
    ) -> Tuple[TensorType[..., 'dim'], TensorType[..., 'dim']]:

        noise_gaussian = torch.randn_like(x)

        if self.is_time_series:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
            noise = L @ noise_gaussian
        else:
            noise = noise_gaussian

        beta_int = self.beta_fn.integral(i)

        mean = x * torch.exp(-beta_int / 2)
        std = (1 - torch.exp(-beta_int)).clamp(1e-5).sqrt()

        y = mean + std * noise

        if _return_all:
            return y, noise, mean, std, cov if self.is_time_series else None

        if self.predict_gaussian_noise:
            return y, noise_gaussian
        else:
            return y, noise

    def get_loss(
        self,
        model: Callable,
        x: TensorType[..., 'dim'],
        **kwargs,
    ) -> TensorType[..., 1]:

        i = torch.rand(x.shape[0], *(1,) * len(x.shape[1:])).expand_as(x[...,:1]).to(x)
        i = i * self.t1

        x_noisy, noise = self.forward(x, i, **kwargs)

        pred_noise = model(x_noisy, i=i, **kwargs)
        loss = self.loss_weighting(i) * (pred_noise - noise)**2

        return loss

    def _get_score(self, model, x, i, L=None, **kwargs):
        """
        Returns score: ∇_xs log p(xs)
        """
        if isinstance(i, float):
            i = torch.Tensor([i]).to(x)
        if i.shape[:-1] != x.shape[:-1]:
            i = i.view(*(1,) * len(x.shape)).expand_as(x[...,:1])

        beta_int = self.beta_fn.integral(i)
        std = (1 - torch.exp(-beta_int)).clamp(1e-5).sqrt()

        noise = model(x, i=i, **kwargs)

        if L is not None:
            # We have to compute the score using -Sigma.inv() @ noise / std
            # assuming noise~N(0, Sigma).
            # If `predict_gaussian_noise=False`, compute (LL^T).inv()
            # Else, we can simplify (LL^T).inv() @ L @ noise
            # to (L^T).inv() @ noise, where noise~N(0, I).
            # So we anyways have to do (L^T).inv(), and sometimes L.inv()
            if not self.predict_gaussian_noise:
                noise = torch.linalg.solve_triangular(L, noise, upper=False)
            noise = torch.linalg.solve_triangular(L.transpose(-1, -2), noise, upper=True)

        score = -noise / std
        return score

    @torch.no_grad()
    def log_prob(
        self,
        model: Callable,
        x: Union[TensorType[..., 'dim'], TensorType[..., 'seq_len', 'dim']],
        num_samples: int = 1,
        **kwargs,
    ) -> TensorType[..., 1]:
        model.train() # Allows backprop through RNN
        self._e = torch.randn(num_samples, *x.shape).to(x)

        if self.is_time_series:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
        else:
            L = None

        def drift(i, state):
            y, _ = state
            with torch.set_grad_enabled(True):
                y = y.requires_grad_(True)
                score = self._get_score(model, y, i=i, L=L, **kwargs)
                if self.is_time_series:
                    # Have to include `cov` since g(t) = "scalar" * L @ dW
                    score = cov @ score
                dy = -0.5 * self.beta_fn(i) * (y + score)
                divergence = divergence_approx(dy, y, self._e, num_samples=num_samples)
            return dy, -divergence

        interval = torch.Tensor([0, self.t1]).to(x)

        # states = odeint(drift, (x, torch.zeros_like(x).to(x)), interval, rtol=1e-6, atol=1e-5)
        states = odeint(drift, (x, torch.zeros_like(x).to(x)), interval,
            method='rk4', options={'step_size': .01})
        y, div = states[0][-1], states[1][-1]

        if self.is_time_series:
            p0 = td.Independent(torch.distributions.MultivariateNormal(
                torch.zeros_like(y).transpose(-1, -2),
                cov.unsqueeze(-3).repeat_interleave(self.dim, dim=-3),
            ), 1)
            log_prob = p0.log_prob(y.transpose(-1, -2)) - div.sum([-1, -2])
            log_prob = log_prob / x.shape[-2]
        else:
            p0 = td.Independent(td.Normal(torch.zeros_like(y), torch.ones_like(y)), 1)
            log_prob = p0.log_prob(y) - div.sum(-1)

        return log_prob.unsqueeze(-1)

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        num_samples: int,
        device: str = None,
        use_ode: bool = True,
        **kwargs,
    ) -> TensorType['num_samples', 'dim']:
        if isinstance(num_samples, int):
            num_samples = (num_samples,)

        sampler = self.ode_sample if use_ode else self.sde_sample
        return sampler(model, num_samples, device, **kwargs)

    @torch.no_grad()
    def ode_sample(
        self,
        model: Callable,
        num_samples: int,
        device: str = None,
        **kwargs,
    ) -> TensorType['num_samples', 'dim']:
        if self.is_time_series:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
        else:
            L = None

        def drift(i, y):
            score = self._get_score(model, y, i=i, L=L, **kwargs)
            if self.is_time_series:
                # Have to include `cov` since g(t) = "scalar" * L @ dW
                score = cov @ score
            return -0.5 * self.beta_fn(i) * (y + score)

        x = self.noise(*num_samples, **kwargs).to(device)
        t = torch.Tensor([self.t1, 0]).to(device)
        y = odeint(drift, x, t, method='rk4', options={'step_size': .01})[1]
        # y = odeint(drift, x, t, rtol=1e-6, atol=1e-5)[1]

        return y

    @torch.no_grad()
    def sde_sample(
        self,
        model: Callable,
        num_samples: int,
        device: str = None,
        **kwargs,
    ) -> TensorType['num_samples', 'dim']:

        if self.is_time_series:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
        else:
            L = None

        is_time_series = self.is_time_series

        x = self.noise(*num_samples, **kwargs).to(device)
        shape = x.shape
        x = x.transpose(-2, -1).flatten(0, -2)

        class SDE(nn.Module):
            noise_type = 'general' if is_time_series else 'diagonal'
            sde_type = 'ito'

            def __init__(self, beta_fn, _get_score):
                super().__init__()
                self.beta_fn = beta_fn
                self._get_score = _get_score

            def f(self, i, inp):
                i = -i
                inp = inp.view(*shape) # Reshape back to original

                score = self._get_score(model, inp, i=i, L=L, **kwargs)
                if is_time_series:
                    score = cov @ score

                dx = self.beta_fn(i) * (0.5 * inp + score)

                if is_time_series:
                    return dx.transpose(-1, -2).flatten(0, -2)
                return dx.view(-1, shape[-1])

            def g(self, i, inp):
                i = -i
                beta = -self.beta_fn(i).sqrt()

                if is_time_series:
                    return (beta * L).repeat_interleave(shape[-1], dim=0)
                return beta.view(1, 1).repeat(np.prod(shape[:-1]), shape[-1]).to(device)

        sde = SDE(self.beta_fn, self._get_score)
        interval = torch.Tensor([-self.t1, 0]).to(device) # Time from -t1 to 0

        step_size = self.t1 / 100
        if not is_time_series:
            x = x.view(-1, shape[-1])
        else:
            x = x.view(-1, shape[-2])
        y = sdeint(sde, x, interval, dt=step_size)[-1]
        y = y.view(*shape)

        return y


class ContinuousGaussianDiffusion(ContinuousDiffusion):
    """ Continuous diffusion using Gaussian noise """
    def __init__(self, dim: int, beta_fn: Callable, predict_gaussian_noise=None, **kwargs):
        super().__init__(dim, beta_fn, noise_fn=Normal(dim), predict_gaussian_noise=True, **kwargs)


class ContinuousOUDiffusion(ContinuousDiffusion):
    """ Continuous diffusion using noise coming from an OU process """
    def __init__(self, dim: int, beta_fn: Callable, predict_gaussian_noise: bool = False, theta: float = 0.5, **kwargs):
        super().__init__(
            dim=dim,
            beta_fn=beta_fn,
            noise_fn=OrnsteinUhlenbeck(dim, theta=theta),
            predict_gaussian_noise=predict_gaussian_noise,
            is_time_series=True,
            **kwargs,
        )


class ContinuousGPDiffusion(ContinuousDiffusion):
    """ Continuous diffusion using noise coming from a Gaussian process """
    def __init__(self, dim: int, beta_fn: Callable, predict_gaussian_noise: bool = False, sigma: float = 0.1, **kwargs):
        super().__init__(
            dim=dim,
            beta_fn=beta_fn,
            noise_fn=GaussianProcess(dim, sigma=sigma),
            predict_gaussian_noise=predict_gaussian_noise,
            is_time_series=True,
            **kwargs,
        )


def divergence_approx(output, input, e, num_samples=1):
    out = 0
    for i in range(num_samples):
        out += torch.autograd.grad(output, input, e[i], create_graph=True)[0].detach() * e[i]
    return out / num_samples




class DiscreteDiffusion(nn.Module):
    """
    Discrete diffusion (https://arxiv.org/abs/2006.11239)

    Args:
        dim: Dimension of data
        num_steps: Number of diffusion steps
        beta_fn: Scheduler for noise levels
        noise_fn: Type of noise
        parallel_elbo: Whether to compute ELBO in parallel or not
    """
    def __init__(
        self,
        dim: int,
        num_steps: int,
        beta_fn: Callable,
        noise_fn: Callable,
        parallel_elbo: bool = False,
        is_time_series: bool = False,
        predict_gaussian_noise: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps
        self.parallel_elbo = parallel_elbo
        self.is_time_series = is_time_series
        self.predict_gaussian_noise = predict_gaussian_noise

        betas = beta_fn(torch.linspace(0, 1, num_steps))
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', torch.cumprod(1 - betas, dim=0))

        self.noise = noise_fn

    def forward(
        self,
        x: TensorType[..., 'dim'],
        i: TensorType[..., 1],
        **kwargs,
    ) -> Tuple[TensorType[..., 'dim'], TensorType[..., 'dim']]:

        noise_gaussian = torch.randn_like(x)

        if self.is_time_series:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
            noise = L @ noise_gaussian
        else:
            noise = noise_gaussian

        alpha = self.alphas[i.long()].to(x)
        y = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise

        if self.predict_gaussian_noise:
            return y, noise_gaussian
        else:
            return y, noise

    def get_loss(
        self,
        model: Callable,
        x: TensorType[..., 'dim'],
        **kwargs,
    ) -> TensorType[..., 'dim']:

        i = torch.randint(0, self.num_steps, size=(x.shape[0],))
        i = i.view(-1, *(1,) * len(x.shape[1:])).expand_as(x[...,:1]).to(x)

        x_noisy, noise = self.forward(x, i, **kwargs)

        pred_noise = model(x_noisy, i=i, **kwargs)
        loss = (pred_noise - noise)**2

        return loss

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        num_samples: Union[int, Tuple],
        device: str = 'cpu',
        **kwargs,
    ) -> TensorType['*num_samples', 'dim']:
        if isinstance(num_samples, int):
            num_samples = (num_samples,)

        x = self.noise(*num_samples, **kwargs).to(device)

        if self.is_time_series and self.predict_gaussian_noise:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
        else:
            L = None

        for diff_step in reversed(range(0, self.num_steps)):
            alpha = self.alphas[diff_step]
            beta = self.betas[diff_step]

            # An alternative can be:
            # alpha_prev = self.alphas[diff_step - 1]
            # sigma = beta * (1 - alpha_prev) / (1 - alpha)
            sigma = beta

            if diff_step == 0:
                z = 0
            else:
                z = self.noise(*num_samples, **kwargs).to(device)

            i = torch.Tensor([diff_step]).expand_as(x[...,:1]).to(device)
            pred_noise = model(x, i=i, **kwargs)

            if L is not None:
                pred_noise = L @ pred_noise

            x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() + sigma.sqrt() * z

        return x

    @torch.no_grad()
    def log_prob(
        self,
        model: Callable,
        x: TensorType[..., 'dim'],
        num_samples: int = 1,
        **kwargs,
    ) -> TensorType[..., 1]:
        if self.is_time_series and self.predict_gaussian_noise:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
        else:
            L = None

        func = self._elbo_parallel if self.parallel_elbo else self._elbo_sequential
        return func(model, x, num_samples=num_samples, L=L, **kwargs)

    def _elbo_parallel(
        self,
        model: Callable,
        x: TensorType[..., 'dim'],
        L: TensorType[..., 'seq_len', 'seq_len'],
        num_samples: int = 1,
        **kwargs,
    ) -> TensorType[..., 1]:
        """
        Computes ELBO over all diffusion steps in parallel,
        then averages over `num_samples` runs.
        If diffusion `num_steps` large (and `num_samples` small)
        it will be heavy on the GPU memory.

        Args:
            model: Denoising diffusion model
            x: Clean input data
            num_samples: How many times to compute ELBO, final
                result is averaged over all ELBO samples
            **kwargs: Can be time, latent etc. depending on a model
        """
        elbo = 0

        i = expand_to_x(torch.arange(self.num_steps), x).expand(-1, *x[...,:1].shape).contiguous()
        alphas = expand_to_x(self.alphas, x)
        betas = expand_to_x(self.betas, x)

        xt, kwargs = expand_x_and_kwargs(x, kwargs, self.num_steps)

        for _ in range(num_samples):
            # Get diffused outputs
            xt, _ = self.forward(x, i, **kwargs) # [num_steps, ..., dim]

            # Output predicted noise
            epsilon = model(xt, i=i, **kwargs)

            if L is not None:
                epsilon = L @ epsilon

            # p(x_{t-1} | p_t)
            p_mu = get_p_mu(xt, betas, alphas, epsilon)
            px = td.Independent(td.Normal(p_mu[1:], betas[1:].sqrt()), 1)

            # p(x_0 | x_1)
            log_prob_x0_x1 = td.Independent(td.Normal(p_mu[0], betas[0].sqrt()), 1).log_prob(x)
            assert log_prob_x0_x1.shape == x.shape[:-1]

            # q(x_{t-1} | x_0, x_t), t > 1
            qx = get_qx(x.unsqueeze(0), xt[1:], alphas[1:], alphas[:-1], betas[1:])

            # KL[q(x_{t-1} | p_t) || p(x_{t-1} | p_t)]
            kl_q_p = td.kl_divergence(qx, px).sum(0)
            assert kl_q_p.shape == x.shape[:-1]

            # ELBO
            elbo_contribution = (log_prob_x0_x1 - kl_q_p) / num_samples
            elbo += elbo_contribution

        elbo = reduce_elbo(elbo, x)
        return elbo

    def _elbo_sequential(
        self,
        model: Callable,
        x: TensorType[..., 'dim'],
        L: TensorType[..., 'seq_len', 'seq_len'],
        num_samples: int = 1,
        **kwargs,
    ) -> TensorType[..., 1]:
        """
        Computes ELBO as a sum of diffusion steps - sequentially.

        Args:
            model: Denoising diffusion model
            x: Clean input data
            num_samples: How many times to compute ELBO, final
                result is averaged over all ELBO samples
            **kwargs: Can be time, latent etc. depending on a model
        """
        elbo = 0

        x, kwargs = expand_x_and_kwargs(x, kwargs, num_samples)

        for i in range(self.num_steps):
            # Prepare variables
            beta = self.betas[i].to(x)
            alpha = self.alphas[i].to(x)
            step = torch.Tensor([i]).expand_as(x[...,:1]).to(x)

            # Diffuse and predict noise
            xt, _ = self.forward(x, i=step, **kwargs)
            epsilon = model(xt, i=step, **kwargs)

            if L is not None:
                epsilon = L @ epsilon

            assert xt.shape == x.shape == epsilon.shape

            # p(x_{t-1} | p_t)
            p_mu = get_p_mu(xt, beta, alpha, epsilon)
            px = td.Independent(td.Normal(p_mu, beta.sqrt()), 1)

            if i == 0:
                elbo = elbo + px.log_prob(x).mean(0)
            else:
                prev_alpha = self.alphas[i - 1]

                # q(x_{t-1} | x_0, x_t), t > 1
                qx = get_qx(x, xt, alpha, prev_alpha, beta)

                # KL[q(x_{t-1} | p_t) || p(x_{t-1} | p_t)]
                kl = td.kl_divergence(qx, px).mean(0)
                elbo = elbo - kl

        elbo = reduce_elbo(elbo, x)
        return elbo


class GaussianDiffusion(DiscreteDiffusion):
    """ Discrete diffusion with Gaussian noise """
    def __init__(self, dim: int, num_steps: int, beta_fn: Callable, **kwargs):
        super().__init__(dim, num_steps, beta_fn, noise_fn=Normal(dim), **kwargs)


class OUDiffusion(DiscreteDiffusion):
    """ Discrete diffusion with noise coming from an OU process """
    def __init__(
        self,
        dim: int,
        num_steps: int,
        beta_fn: Callable,
        predict_gaussian_noise: bool,
        theta: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            num_steps=num_steps,
            beta_fn=beta_fn,
            noise_fn=OrnsteinUhlenbeck(dim, theta=theta),
            is_time_series=True,
            predict_gaussian_noise=predict_gaussian_noise,
            **kwargs,
        )


class GPDiffusion(DiscreteDiffusion):
    """ Discrete diffusion with noise coming from a Gaussian process """
    def __init__(
        self,
        dim: int,
        num_steps: int,
        beta_fn: Callable,
        predict_gaussian_noise: bool,
        sigma: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            num_steps=num_steps,
            beta_fn=beta_fn,
            noise_fn=GaussianProcess(dim, sigma=sigma),
            is_time_series=True,
            predict_gaussian_noise=predict_gaussian_noise,
            **kwargs,
        )


def expand_to_x(inputs, x):
    return inputs.view(-1, *(1,) * len(x.shape)).to(x)

def expand_x_and_kwargs(x, kwargs, N):
    # Expand dimensions
    x = x.unsqueeze(0).repeat_interleave(N, dim=0)

    # A hacky solution to repeat dimensions in all kwargs (latent, t, etc.)
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            kwargs[key] = value.unsqueeze(0).repeat_interleave(N, dim=0)

    return x, kwargs

def reduce_elbo(
    elbo: TensorType['batch', Any],
    x: TensorType[Any],
) -> TensorType['batch', 1]:
    # Reduce ELBO over all but batch dimension: (B, ...) -> (B,)
    elbo = elbo.view(elbo.shape[0], -1).sum(1)

    if len(x.shape) > 2:
        elbo = elbo / x.shape[-2]

    return elbo.unsqueeze(1)

def get_p_mu(xt, beta, alpha, epsilon):
    mu = 1 / (1 - beta).sqrt() * (xt - beta / (1 - alpha).sqrt() * epsilon)
    return mu

def get_qx(x, xt, alpha, prev_alpha, beta):
    q_mu_1 = torch.sqrt(prev_alpha) * beta / (1 - alpha) * x
    q_mu_2 = torch.sqrt(1 - beta) * (1 - prev_alpha) / (1 - alpha) * xt
    q_mu = q_mu_1 + q_mu_2

    q_sigma = beta * (1 - prev_alpha) / (1 - alpha)

    qx = td.Independent(td.Normal(q_mu, q_sigma.expand_as(q_mu).sqrt()), 1)
    return qx