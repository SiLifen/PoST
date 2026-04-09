"""
Mamba-2 PoST Mixer for Zoology MQAR Benchmark.

Wraps Mamba2PoST from the PoST project as a Zoology-compatible
sequence mixer. Provides both:
  - Mamba2PoSTMixer: the sequence mixer module
  - Mamba2PoSTBlock: a custom block type (RMSNorm + mixer + residual, no MLP)
"""

import torch
import torch.nn as nn

# Import from the PoST project (parent directory)
import sys
import os

# Add PoST root to path so we can import models
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.post_mamba2 import Mamba2PoSTConfig, Mamba2PoST


class Mamba2PoSTMixer(nn.Module):
    """
    Zoology-compatible wrapper for Mamba2PoST.

    Interface: __init__(d_model, layer_idx, **kwargs) -> forward(hidden_states) -> hidden_states

    Key kwargs (passed through to Mamba2PoST):
        d_state:           SSM state dimension
        headdim:           head dimension
        train_length:      training context length (determines geometric init range)
        use_post:          whether to enable PoST (default True)
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int = None,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
        train_length: int = 2048,
        use_post: bool = True,
        **kwargs,  # absorb any extra kwargs from experiment configs
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
            train_length=train_length,
            use_post=use_post,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states):
        return self.mixer(hidden_states)

    def state_size(self, sequence_length: int = 2048):
        """Report state size for Zoology's memory tracking."""
        return self.mixer.intermediate_size * self.mixer.ssm_state_size


class RMSNorm(nn.Module):
    """Simple RMSNorm matching PoST's native normalization."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba2PoSTBlock(nn.Module):
    """
    Custom block type for Zoology that matches PoST's native architecture:
      residual + RMSNorm -> Mamba2PoSTMixer -> output

    No MLP, no LayerNorm — just RMSNorm + SSM + residual, exactly matching
    Mamba2PoSTForCausalLM.forward().
    """

    def __init__(self, config, layer_idx: int = None, **kwargs):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mixer = Mamba2PoSTMixer(
            d_model=config.d_model,
            layer_idx=layer_idx,
            **config.sequence_mixer.kwargs,
        )

    def forward(self, hidden_states, residual=None):
        """
        Zoology block interface: (hidden_states, residual) -> (hidden_states, residual)
        """
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual
