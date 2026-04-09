"""
Mamba-2 Mixer for Zoology MQAR Benchmark.

Uses Mamba2PoST from the PoST project with use_post=False for vanilla
Mamba-2 behavior. This ensures we use the same high-quality fla-based
implementation (with causal_conv1d_fn, proper fused kernels, etc.)
instead of a separate standalone re-implementation.
"""

import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor

# Import from the PoST project (parent directory)
import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.post_mamba2 import Mamba2PoST


class Mamba2(nn.Module):
    """
    Zoology-compatible Mamba-2 mixer backed by Mamba2PoST(use_post=False).

    Delegates to fla's Mamba2 implementation which uses causal_conv1d_fn
    and properly fused kernels.

    Interface: __init__(d_model, ...) -> forward(hidden_states) -> hidden_states
    """

    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        chunk_size=256,
        layer_idx=None,
        **kwargs,  # absorb extra kwargs (use_fast_path, num_feature_maps, etc.)
    ):
        super().__init__()

        num_heads = int(expand * d_model) // headdim

        self.mixer = Mamba2PoST(
            num_heads=num_heads,
            head_dim=headdim,
            hidden_size=d_model,
            state_size=d_state,
            expand=expand,
            conv_kernel=d_conv,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
            use_post=False,  # vanilla Mamba-2
        )

        self.d_model = d_model
        self.d_state = d_state

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim)
        Returns: same shape as u
        """
        return self.mixer(u)

    def state_size(self, sequence_length: int = 2048):
        return self.mixer.intermediate_size * self.mixer.ssm_state_size


class RMSNorm(nn.Module):
    """Simple RMSNorm matching PoST's native normalization."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba2Block(nn.Module):
    """
    Block type for Zoology: RMSNorm -> Mamba2 mixer -> residual.
    No MLP — matches Mamba-2's native architecture.
    """

    def __init__(self, config, layer_idx: int = None, **kwargs):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mixer = Mamba2(
            d_model=config.d_model,
            layer_idx=layer_idx,
            **config.sequence_mixer.kwargs,
        )

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None,
        inference_params=None, **mixer_kwargs
    ):
        """
        Zoology block interface: (hidden_states, residual) -> (hidden_states, residual)
        """
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual
