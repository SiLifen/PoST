# -*- coding: utf-8 -*-
# RetNet (Multi-Scale Retention) mixer for Zoology MQAR benchmark.
#
# Wraps fla.layers.multiscale_retention.MultiScaleRetention with a
# Zoology-compatible interface (d_model kwarg, single-tensor output).

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from fla.layers.multiscale_retention import MultiScaleRetention

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


class RetNet(nn.Module):
    """
    Zoology-compatible wrapper for Multi-Scale Retention (RetNet).

    Reference: "Retentive Network: A Successor to Transformer for Large Language Models"
               (https://arxiv.org/abs/2307.08621)

    Args:
        d_model (int): Hidden size. Default: 256.
        num_heads (int): Number of attention heads. Default: 4.
        expand_k (float): Key expansion ratio. Default: 1.0.
        expand_v (float): Value expansion ratio. Default: 2.0.
        use_short_conv (bool): Whether to use short convolutions. Default: True.
        conv_size (int): Short convolution kernel size. Default: 4.
        use_output_gate (bool): Whether to use output gate. Default: True.
        mode (str): Kernel mode. Default: 'chunk'.
        layer_idx (int): Layer index. Default: None.
        norm_eps (float): Norm epsilon. Default: 1e-5.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        mode: str = 'chunk',
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()

        self.layer = MultiScaleRetention(
            mode=mode,
            hidden_size=d_model,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
            use_output_gate=use_output_gate,
            gate_fn=gate_fn,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
        )

        self.hidden_size = d_model
        self.num_heads = num_heads
        self.head_k_dim = int(d_model * expand_k) // num_heads
        self.head_v_dim = int(d_model * expand_v) // num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        # MultiScaleRetention returns (o, attn_weights, past_key_values)
        o, _, _ = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        return o

    def state_size(self, sequence_length: int = 2048):
        return self.num_heads * self.head_k_dim * self.head_v_dim
