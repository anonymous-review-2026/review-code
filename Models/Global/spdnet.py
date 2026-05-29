from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class StiefelParameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True):
        return super().__new__(cls, data, requires_grad=requires_grad)


def orthogonal_projection(grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    wt_g = weight.transpose(-1, -2) @ grad
    sym = 0.5 * (wt_g + wt_g.transpose(-1, -2))
    return grad - weight @ sym


def retraction(weight: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(weight)
    diag = torch.diag(r, 0)
    phase = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
    q *= phase
    return q


class StiefelMetaOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.grad[torch.isnan(param.grad)] = 0.0
                if isinstance(param, StiefelParameter):
                    param.grad.copy_(orthogonal_projection(param.grad, param))

        loss = self.optimizer.step(closure)
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if isinstance(param, StiefelParameter):
                    param.data.copy_(retraction(param.data))
        return loss


def _safe_eigh(x: torch.Tensor, jitter: float = 1e-4, max_tries: int = 6):
    if x.ndim != 3 or x.shape[-1] != x.shape[-2]:
        raise ValueError(f"_safe_eigh expects (B,n,n), got shape={tuple(x.shape)}")

    x0_dtype = x.dtype
    x = 0.5 * (x + x.transpose(-1, -2))
    x64 = x.to(torch.float64)
    _, n, _ = x64.shape
    eye = torch.eye(n, device=x64.device, dtype=x64.dtype).unsqueeze(0)

    jit = float(jitter)
    last_err = None
    for _ in range(max_tries):
        try:
            s, u = torch.linalg.eigh(x64 + jit * eye)
            return s.to(x0_dtype), u.to(x0_dtype)
        except RuntimeError as e:
            last_err = e
            jit *= 10.0
    raise last_err


class SPDTransform(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        if input_size < output_size:
            raise ValueError(f"SPDTransform requires input_size >= output_size, got {input_size} < {output_size}")
        self.weight = StiefelParameter(torch.empty(input_size, output_size), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        return w.transpose(-2, -1) @ x @ w


class SPDRectified(nn.Module):
    def __init__(self, epsilon: float = 1e-4):
        super().__init__()
        self.register_buffer("epsilon", torch.tensor([epsilon], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, u = _safe_eigh(x, jitter=float(self.epsilon[0]))
        s = s.clamp(min=float(self.epsilon[0]))
        return u @ s.diag_embed() @ u.transpose(-2, -1)


class SPDTangentSpace(nn.Module):
    def __init__(self, vectorization: bool = True):
        super().__init__()
        self.vectorization = vectorization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, u = _safe_eigh(x, jitter=1e-6)
        s = s.clamp(min=1e-12).log().diag_embed()
        out = u @ s @ u.transpose(-2, -1)

        if not self.vectorization:
            return torch.flatten(out, 1)

        rows, cols = torch.triu_indices(out.shape[1], out.shape[2], offset=0, device=out.device)
        vec = out[:, rows, cols]
        scale = torch.ones(rows.numel(), dtype=out.dtype, device=out.device)
        scale[rows != cols] = math.sqrt(2.0)
        return vec * scale


class BrainSPDFeaturizer(nn.Module):
    def __init__(self, input_rois: int, hidden_sizes: Sequence[int] = (64, 32), epsilon: float = 1e-4):
        super().__init__()
        sizes = [int(input_rois), *[int(s) for s in hidden_sizes]]
        layers = []
        for in_s, out_s in zip(sizes[:-1], sizes[1:]):
            layers.append(SPDTransform(in_s, out_s))
            layers.append(SPDRectified(epsilon=epsilon))
        layers.append(SPDTangentSpace(vectorization=True))
        self.layers = nn.Sequential(*layers)
        final_n = sizes[-1]
        self.out_dim = int(final_n * (final_n + 1) / 2)

    def forward(self, spd: torch.Tensor) -> torch.Tensor:
        return self.layers(spd)
