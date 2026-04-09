# -*- coding: utf-8 -*-
"""
Gated DeltaNet mixer with PoST modifications for Zoology MQAR.

Wraps the unified GDNPoST from models.post_gated_deltanet as a
Zoology-compatible sequence mixer. Inherits from fla's GatedDeltaNet;
when use_post=False behaves identically to the vanilla baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.post_gated_deltanet import GatedDeltaNetConfig, GDNPoST


class GatedDeltaNetPoST(nn.Module):
    """
    Zoology-compatible wrapper for GDNPoST.

    Interface: __init__(d_model, ...) -> forward(hidden_states) -> hidden_states

    Key kwargs:
        num_heads:       number of attention heads
        expand_v:        value expansion factor
        use_gate:        whether to use output gate
        use_short_conv:  whether to use short convolutions
        conv_size:       convolution kernel size
        train_length:    training context length (for PoST geometric init)
        use_post:        whether to enable PoST (default True)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 2,
        expand_v: float = 2,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        train_length: int = 512,
        use_post: bool = True,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()

        config = GatedDeltaNetConfig(
            d_model=d_model,
            n_layer=1,
            num_heads=num_heads,
            expand_v=expand_v,
            use_gate=use_gate,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            vocab_size=1,
            train_length=train_length,
            norm_eps=norm_eps,
            use_post=use_post,
        )

        self.mixer = GDNPoST(config, layer_idx=layer_idx or 0)
        self.num_heads = num_heads
        self.head_k_dim = d_model // num_heads
        self.head_v_dim = int(self.head_k_dim * expand_v)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        o, _, _ = self.mixer(
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
