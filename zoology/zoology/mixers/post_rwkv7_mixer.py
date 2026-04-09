"""
RWKV-7 PoST Mixer for Zoology MQAR Benchmark.

Wraps RWKV7PoSTAttention from the PoST project as a Zoology-compatible
sequence mixer. Provides both:
  - RWKV7PoSTMixer: the sequence mixer module
  - RWKV7PoSTBlock: a custom block type (RMSNorm + mixer + residual, no MLP)

Usage in experiment configs:
    mixer = dict(
        name="zoology.mixers.post_rwkv7_mixer.RWKV7PoSTMixer",
        kwargs={"post_mode": "adaptive", "head_dim": 16},
    )
    model = ModelConfig(
        block_type="RWKV7PoSTBlock",
        ...
    )
"""

import torch
import torch.nn as nn

# Import from the PoST project (parent directory)
import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.post_rwkv7 import RWKV7Config, RWKV7PoSTAttention


class RWKV7PoSTMixer(nn.Module):
    """
    Zoology-compatible wrapper for RWKV7PoSTAttention.

    Interface: __init__(d_model, layer_idx, **kwargs) -> forward(hidden_states) -> hidden_states

    Key kwargs (passed through to RWKV7Config):
        head_dim:          per-head dimension (d_model must be divisible)
        train_length:      training context length (geometric init range)
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int = None,
        head_dim: int = 16,
        n_layers: int = 2,
        train_length: int = 512,
        use_post: bool = True,
        decay_low_rank_dim: int = None,
        gate_low_rank_dim: int = None,
        a_low_rank_dim: int = None,
        v_low_rank_dim: int = None,
        **kwargs,
    ):
        super().__init__()

        config = RWKV7Config(
            d_model=d_model,
            n_layer=n_layers,
            head_dim=head_dim,
            vocab_size=1,
            train_length=train_length,
            use_post=use_post,
            decay_low_rank_dim=decay_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            a_low_rank_dim=a_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
        )

        self.mixer = RWKV7PoSTAttention(config, layer_idx=layer_idx or 0)

    def forward(self, hidden_states, v_first=None):
        # Check batch size compatibility
        if v_first is not None and v_first.shape[0] != hidden_states.shape[0]:
            v_first = None
        out, _, _, v_first_out = self.mixer(
            hidden_states, v_first=v_first
        )
        return out, v_first_out

    def state_size(self, sequence_length: int = 2048):
        """Report state size for Zoology's memory tracking."""
        return self.mixer.num_heads * self.mixer.head_dim * self.mixer.head_dim





class RWKV7PoSTBlock(nn.Module):
    """
    Custom block type for Zoology that matches RWKV-7's native architecture:
      residual + RMSNorm -> RWKV7PoSTMixer -> output

    No MLP, no LayerNorm — just RMSNorm + mixer + residual.
    """

    def __init__(self, config, layer_idx: int = None, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mixer = RWKV7PoSTMixer(
            d_model=config.d_model,
            layer_idx=layer_idx,
            n_layers=config.n_layers,
            **config.sequence_mixer.kwargs,
        )

    def forward(self, hidden_states, residual=None, v_first=None):
        """
        Zoology block interface extended with v_first:
        (hidden_states, residual, v_first) -> (hidden_states, residual, v_first)
        """
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual)
        hidden_states, v_first = self.mixer(hidden_states, v_first=v_first)
        return hidden_states, residual, v_first
