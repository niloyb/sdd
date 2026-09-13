"""Synthetic generators with tractable conditional laws.

A generator draws series and reports the conditional mean/std of the future given a
prefix -- the second is what SDD trains against. Every conditional here is Gaussian so
the distilled losses apply directly; `GeometricBrownianMotion` works in log space to
keep that true.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

Tensor = torch.Tensor


class Generator(ABC):
    """A data-generating process whose conditional law is available in closed form."""

    name: str

    @abstractmethod
    def sample(self, n: int, T: int, g: torch.Generator) -> tuple[Tensor, Tensor]:
        """Draw `n` series of length `T`.

        `params` carries whatever `posterior` needs to rebuild that series' law, cached
        alongside `y` so posteriors are never stored at O(T^2).
        """

    @abstractmethod
    def posterior(self, y: Tensor, params: Tensor, ctx: int) -> tuple[Tensor, Tensor]:
        """Mean and std of `y[:, ctx:]` given `y[:, :ctx]`, per time point."""


# --------------------------------------------------------------------------- GP

_KERNELS = (
    "rbf",
    "matern12",
    "matern32",
    "matern52",
    "periodic",
    "rq",
    "locally_periodic",
    "linear",
    "poly2",
    "spectral",
)

# params layout: [kernel_id, lengthscale, outputscale, period, rq_alpha,
#                 periodic_ls, poly_offset, slope, intercept, noise]
_GP_P = 10


def _gram(kid: int, p: Tensor, u: Tensor) -> Tensor:
    """Kernel matrix on grid `u` for one series' hyperparameters."""
    d = (u[:, None] - u[None, :]).abs()
    ls, os_, per, alpha, pls = p[1], p[2], p[3], p[4], p[5]
    name = _KERNELS[kid]
    if name == "rbf":
        k = torch.exp(-0.5 * (d / ls) ** 2)
    elif name == "matern12":
        k = torch.exp(-d / ls)
    elif name == "matern32":
        r = math.sqrt(3.0) * d / ls
        k = (1 + r) * torch.exp(-r)
    elif name == "matern52":
        r = math.sqrt(5.0) * d / ls
        k = (1 + r + r**2 / 3) * torch.exp(-r)
    elif name == "periodic":
        k = torch.exp(-2 * torch.sin(math.pi * d / per) ** 2 / pls**2)
    elif name == "rq":
        k = (1 + d**2 / (2 * alpha * ls**2)) ** (-alpha)
    elif name == "locally_periodic":
        k = torch.exp(-2 * torch.sin(math.pi * d / per) ** 2 / pls**2) * torch.exp(
            -0.5 * (d / ls) ** 2
        )
    elif name == "linear":
        k = u[:, None] * u[None, :] / ls**2
    elif name == "poly2":
        k = (u[:, None] * u[None, :] / ls**2 + p[6]) ** 2
    else:  # spectral: three fixed-weight components
        k = torch.zeros_like(d)
        for w, f in ((1.0, 0.01), (0.5, 0.05), (0.25, 0.15)):
            k = k + w * torch.exp(-0.5 * (d * f * ls) ** 2) * torch.cos(
                2 * math.pi * f * d
            )
        k = k / 1.75
    return os_**2 * k


class GaussianProcess(Generator):
    """GP draws with a per-series kernel, hyperparameters and observation noise."""

    name = "gp"

    def __init__(self, noise_std: float = 0.25, kernels: tuple[str, ...] = _KERNELS):
        self.noise_std = noise_std
        self.kernel_ids = [_KERNELS.index(k) for k in kernels]

    def _u(self, T: int, device, dtype) -> Tensor:
        return torch.arange(T, device=device, dtype=dtype)

    def sample(self, n: int, T: int, g: torch.Generator) -> tuple[Tensor, Tensor]:
        dev, dt = g.device, torch.float64
        p = torch.empty(n, _GP_P, device=dev, dtype=dt)
        pick = torch.tensor(self.kernel_ids, device=dev)
        p[:, 0] = pick[torch.randint(len(pick), (n,), generator=g, device=dev)]
        p[:, 1] = 5 + 45 * torch.rand(n, generator=g, device=dev, dtype=dt)
        p[:, 2] = 0.5 + 1.5 * torch.rand(n, generator=g, device=dev, dtype=dt)
        p[:, 3] = 8 + 56 * torch.rand(n, generator=g, device=dev, dtype=dt)
        p[:, 4] = 0.5 + 3.5 * torch.rand(n, generator=g, device=dev, dtype=dt)
        p[:, 5] = 0.5 + 1.5 * torch.rand(n, generator=g, device=dev, dtype=dt)
        p[:, 6] = 2 * torch.rand(n, generator=g, device=dev, dtype=dt)
        linear = torch.rand(n, generator=g, device=dev, dtype=dt) < 0.5
        p[:, 7] = torch.where(
            linear, 0.04 * torch.rand(n, generator=g, device=dev, dtype=dt) - 0.02, 0.0
        )
        p[:, 8] = 2 * torch.rand(n, generator=g, device=dev, dtype=dt) - 1
        p[:, 9] = self.noise_std

        u = self._u(T, dev, dt)
        y = torch.empty(n, T, device=dev, dtype=dt)
        mean = p[:, 7:8] * u[None, :] + p[:, 8:9]
        for i in range(n):  # per-series kernel; batched Cholesky would need equal kids
            k = _gram(int(p[i, 0]), p[i], u)
            k = k + (p[i, 9] ** 2 + 1e-4) * torch.eye(T, device=dev, dtype=dt)
            L = torch.linalg.cholesky(k)
            xi = torch.randn(T, generator=g, device=dev, dtype=dt)
            y[i] = mean[i] + L @ xi
        return y.float(), p.float()

    def posterior(self, y: Tensor, params: Tensor, ctx: int) -> tuple[Tensor, Tensor]:
        n, T = y.shape
        dev, dt = y.device, torch.float64
        u = self._u(T, dev, dt)
        mu = torch.empty(n, T - ctx, device=dev, dtype=dt)
        sd = torch.empty(n, T - ctx, device=dev, dtype=dt)
        yd, pd = y.to(dt), params.to(dt)
        for i in range(n):
            p = pd[i]
            k = _gram(int(p[0]), p, u) + (p[9] ** 2 + 1e-4) * torch.eye(
                T, device=dev, dtype=dt
            )
            m = p[7] * u + p[8]
            kcc, kqc, kqq = k[:ctx, :ctx], k[ctx:, :ctx], k[ctx:, ctx:]
            sol = torch.linalg.solve(kcc, (yd[i, :ctx] - m[:ctx]).unsqueeze(-1))
            mu[i] = m[ctx:] + (kqc @ sol).squeeze(-1)
            v = kqq.diagonal() - (kqc @ torch.linalg.solve(kcc, kqc.T)).diagonal()
            sd[i] = v.clamp_min(1e-12).sqrt()
        return mu.float(), sd.float()


# ------------------------------------------------------- linear Gaussian SSM

class LinearGaussianSSM(Generator):
    """x_{t+1} = a x_t + w, y_t = x_t + v. Covers AR(1)/DLM-style generators."""

    name = "ssm"

    # params: [a, sigma_w, sigma_v, x0]
    def sample(self, n: int, T: int, g: torch.Generator) -> tuple[Tensor, Tensor]:
        dev = g.device
        a = 0.8 + 0.19 * torch.rand(n, generator=g, device=dev)
        sw = 0.1 + 0.4 * torch.rand(n, generator=g, device=dev)
        sv = 0.1 + 0.4 * torch.rand(n, generator=g, device=dev)
        x = torch.randn(n, generator=g, device=dev)
        p = torch.stack([a, sw, sv, x], dim=1)
        ys = []
        for _ in range(T):
            x = a * x + sw * torch.randn(n, generator=g, device=dev)
            ys.append(x + sv * torch.randn(n, generator=g, device=dev))
        return torch.stack(ys, dim=1), p

    def posterior(self, y: Tensor, params: Tensor, ctx: int) -> tuple[Tensor, Tensor]:
        a, sw, sv = params[:, 0], params[:, 1], params[:, 2]
        m = torch.zeros_like(a)
        v = sw**2 / (1 - a**2).clamp_min(1e-6)
        for t in range(ctx):  # Kalman filter over the observed prefix
            m, v = a * m, a**2 * v + sw**2
            k = v / (v + sv**2)
            m = m + k * (y[:, t] - m)
            v = (1 - k) * v
        mus, sds = [], []
        for _ in range(y.shape[1] - ctx):
            m, v = a * m, a**2 * v + sw**2
            mus.append(m)
            sds.append((v + sv**2).sqrt())
        return torch.stack(mus, 1), torch.stack(sds, 1)


# ------------------------------------------------------------------- SDEs

class OrnsteinUhlenbeck(Generator):
    """dX = -theta X dt + sigma dW, observed on the integer grid."""

    name = "ou"

    # params: [theta, sigma, x0]
    def sample(self, n: int, T: int, g: torch.Generator) -> tuple[Tensor, Tensor]:
        dev = g.device
        th = 0.02 + 0.18 * torch.rand(n, generator=g, device=dev)
        sg = 0.5 + 1.5 * torch.rand(n, generator=g, device=dev)
        x = sg / (2 * th).sqrt() * torch.randn(n, generator=g, device=dev)
        p = torch.stack([th, sg, x], dim=1)
        d = torch.exp(-th)
        sd = sg * ((1 - d**2) / (2 * th)).sqrt()
        ys = []
        for _ in range(T):
            x = d * x + sd * torch.randn(n, generator=g, device=dev)
            ys.append(x)
        return torch.stack(ys, dim=1), p

    def posterior(self, y: Tensor, params: Tensor, ctx: int) -> tuple[Tensor, Tensor]:
        th, sg = params[:, 0], params[:, 1]
        x = y[:, ctx - 1]  # Markov: the last observation is the state
        mus, sds = [], []
        for k in range(1, y.shape[1] - ctx + 1):
            mus.append(x * torch.exp(-th * k))
            sds.append((sg**2 / (2 * th) * (1 - torch.exp(-2 * th * k))).sqrt())
        return torch.stack(mus, 1), torch.stack(sds, 1)


class GeometricBrownianMotion(OrnsteinUhlenbeck):
    """GBM in log space, where conditionals stay Gaussian. Exponentiate for prices."""

    name = "gbm"

    # params: [mu, sigma, logx0]
    def sample(self, n: int, T: int, g: torch.Generator) -> tuple[Tensor, Tensor]:
        dev = g.device
        mu = 0.02 * torch.randn(n, generator=g, device=dev)
        sg = 0.05 + 0.15 * torch.rand(n, generator=g, device=dev)
        x = torch.zeros(n, device=dev)
        p = torch.stack([mu, sg, x], dim=1)
        ys = []
        for _ in range(T):
            x = x + (mu - 0.5 * sg**2) + sg * torch.randn(n, generator=g, device=dev)
            ys.append(x)
        return torch.stack(ys, dim=1), p

    def posterior(self, y: Tensor, params: Tensor, ctx: int) -> tuple[Tensor, Tensor]:
        mu, sg = params[:, 0], params[:, 1]
        x = y[:, ctx - 1]
        mus, sds = [], []
        for k in range(1, y.shape[1] - ctx + 1):
            mus.append(x + k * (mu - 0.5 * sg**2))
            sds.append(sg * math.sqrt(k) * torch.ones_like(x))
        return torch.stack(mus, 1), torch.stack(sds, 1)


# ------------------------------------------------------------------- i.n.i.d.

class INID(Generator):
    """y_t = g(t) + eps_t, with trend, seasonality and a change point.

    The future is independent of the history, so the conditional law is the marginal.
    """

    name = "inid"

    # params: [level, slope, amp, period, phase, noise, cp_time, cp_jump]
    def sample(self, n: int, T: int, g: torch.Generator) -> tuple[Tensor, Tensor]:
        dev = g.device
        r = lambda a, b: a + (b - a) * torch.rand(n, generator=g, device=dev)
        p = torch.stack(
            [
                r(-1, 1),
                r(-0.02, 0.02),
                r(0, 2),
                r(8, 64),
                r(0, 2 * math.pi),
                r(0.1, 0.5),
                r(0.2, 0.8) * T,
                r(-2, 2),
            ],
            dim=1,
        )
        t = torch.arange(T, device=dev, dtype=torch.float32)
        m = self._mean(p, t)
        return m + p[:, 5:6] * torch.randn(n, T, generator=g, device=dev), p

    @staticmethod
    def _mean(p: Tensor, t: Tensor) -> Tensor:
        base = p[:, 0:1] + p[:, 1:2] * t[None, :]
        seas = p[:, 2:3] * torch.sin(2 * math.pi * t[None, :] / p[:, 3:4] + p[:, 4:5])
        jump = p[:, 7:8] * (t[None, :] >= p[:, 6:7]).float()
        return base + seas + jump

    def posterior(self, y: Tensor, params: Tensor, ctx: int) -> tuple[Tensor, Tensor]:
        t = torch.arange(y.shape[1], device=y.device, dtype=torch.float32)[ctx:]
        mu = self._mean(params, t)
        sd = params[:, 5:6].expand_as(mu)
        return mu, sd


REGISTRY: dict[str, type[Generator]] = {
    g.name: g
    for g in (
        GaussianProcess,
        LinearGaussianSSM,
        OrnsteinUhlenbeck,
        GeometricBrownianMotion,
        INID,
    )
}
