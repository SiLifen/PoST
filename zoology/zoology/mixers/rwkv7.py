# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Vanilla RWKV-7 mixer for Zoology, using the official RWKV-Vibe/rwkv-fla kernels.
# This is the baseline (non-PoST) RWKV-7 implementation.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.rwkv6 import LoRA
from fla.modules.token_shift import token_shift
from fla.ops.rwkv7 import chunk_rwkv7, fused_mul_recurrent_rwkv7
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7
from fla.ops.rwkv7.gate_output_correction import gate_output_correction

if TYPE_CHECKING:
    from fla.models.utils import Cache


class RWKV7Attention(nn.Module):
    """
    Vanilla RWKV-7 attention layer for Zoology benchmarks.

    Follows the official RWKV-Vibe/rwkv-fla implementation exactly:
    - Separate x_r/x_w/x_k/x_v/x_a/x_g token-shift parameters
    - fused_addcmul_rwkv7 for token-shift mixing
    - fused_k_rwkv7 for key update
    - gate_output_correction for gated output
    - token_shift from fla.modules
    - Official RWKV-7 weight initialization
    """

    def __init__(
        self,
        mode: str = 'chunk',
        d_model: int = 1024,
        head_dim: Optional[int] = 64,
        num_heads: Optional[int] = None,
        decay_low_rank_dim: int = None,
        gate_low_rank_dim: int = None,
        a_low_rank_dim: int = None,
        v_low_rank_dim: int = None,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        num_hidden_layers: int = 2,
        **kwargs
    ) -> RWKV7Attention:
        super().__init__()

        self.mode = mode
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        hidden_size = int(d_model)
        self.hidden_size = hidden_size

        self.key_dim = hidden_size
        self.value_dim = hidden_size
        if head_dim is not None and num_heads is not None:
            raise ValueError("Cannot specify both `head_dim` and `num_heads`.")
        elif head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = int(hidden_size // head_dim)
        elif num_heads is not None:
            self.head_dim = int(hidden_size // num_heads)
            self.num_heads = num_heads
        else:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")
        self.head_v_dim = hidden_size // self.num_heads

        # LoRA rank calculation (official defaults)
        factor = self.head_dim / 64
        if decay_low_rank_dim is None:
            decay_low_rank_dim = max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
        if gate_low_rank_dim is None:
            gate_low_rank_dim = max(32, int(round((5 * (hidden_size**0.5)) / 32) * 32))
        if a_low_rank_dim is None:
            a_low_rank_dim = max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
        if v_low_rank_dim is None:
            v_low_rank_dim = max(32, int(round((1.7 * (hidden_size**0.5)) * factor / 32) * 32))

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # Separate token-shift parameters (official RWKV-7 style)
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.w_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=decay_low_rank_dim, activation='tanh')
        if self.layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=v_low_rank_dim, activation=None)
        self.a_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=a_low_rank_dim, activation=None)
        self.g_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=gate_low_rank_dim, activation='sigmoid', bias=False)

        self.g_norm = nn.GroupNorm(
            num_groups=self.num_heads,
            num_channels=self.value_dim,
            eps=self.head_dim * norm_eps,
            affine=elementwise_affine
        )

        self._initialize_rwkv7_weights()

    @torch.no_grad()
    @torch.compiler.disable
    def _initialize_rwkv7_weights(self):
        """Official RWKV-7 weight initialization (from rwkv-fla)."""
        hidden_size = self.hidden_size
        layer_idx = self.layer_idx if self.layer_idx is not None else 0
        n_layer = self.num_hidden_layers

        if n_layer <= 1:
            ratio_0_to_1 = 0.0
            ratio_1_to_almost0 = 1.0
        else:
            ratio_0_to_1 = layer_idx / (n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_idx / n_layer)

        ddd = torch.ones(1, 1, hidden_size)
        www = torch.zeros(hidden_size)
        zigzag = torch.zeros(hidden_size)
        linear = torch.zeros(hidden_size)
        for n in range(hidden_size):
            linear[n] = n / max(hidden_size - 1, 1) - 0.5
            zigzag[n] = ((n % self.head_dim) - ((self.head_dim - 1) / 2)) / max((self.head_dim - 1) / 2, 1)
            zigzag[n] = zigzag[n] * abs(zigzag[n])
            www[n] = -6 + 6 * (n / max(hidden_size - 1, 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)
            ddd[0, 0, n] = n / hidden_size

        # Token-shift mixing
        self.x_r.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_r.dtype)
        self.x_w.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_w.dtype)
        self.x_k.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).to(self.x_k.dtype)
        self.x_v.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).to(self.x_v.dtype)
        self.x_a.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_a.dtype)
        self.x_g.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_g.dtype)

        # k_k, k_a, r_k
        nn.init.constant_(self.k_a, 1.02)
        nn.init.constant_(self.r_k, -0.04)
        self.k_k.data.copy_((torch.zeros(hidden_size) + 0.71 - linear * 0.1).to(self.k_k.dtype))

        # LoRA biases
        self.w_lora.set_bias_value(www + 0.5 + zigzag * 2.5)
        self.a_lora.set_bias_value(-0.19 + zigzag * 0.3 + linear * 0.4)
        if layer_idx != 0:
            self.v_lora._initialize_weights(self.v_lora)
            self.v_lora.set_bias_value(0.73 - linear * 0.4)

        # GroupNorm
        if n_layer > 0:
            self.g_norm.weight.data[:] = ((layer_idx + 1) / n_layer) ** 0.7

        # Linear projections (orthogonal)
        self._orthogonal_init(self.r_proj.weight)
        self._orthogonal_init(self.k_proj.weight, gain=0.1)
        self._orthogonal_init(self.v_proj.weight)
        self.o_proj.weight.data.zero_()

        # Protect all custom-initialized params from zoology's _init_weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight._no_reinit = True
                if module.bias is not None:
                    module.bias._no_reinit = True
        for p in self.parameters():
            p._no_reinit = True

    @staticmethod
    def _orthogonal_init(weight, gain=1.0):
        original_dtype = weight.dtype
        weight_f = weight.float()
        nn.init.orthogonal_(weight_f, gain=gain)
        weight.data.copy_(weight_f.to(original_dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        v_first: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2
            am = attention_mask.narrow(1, attention_mask.size(1) - seq_len, seq_len).unsqueeze(-1)

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if attention_mask is not None:
            hidden_states = hidden_states.mul(am)

        if last_state is None:
            conv_cache = None
            recurrent_state = None
        else:
            conv_cache = last_state['conv_state']
            recurrent_state = last_state['recurrent_state']

        cu_seqlens = kwargs.get('cu_seqlens', None)

        # Token-shift (official fused op)
        delta, conv_state = token_shift(
            hidden_states, cu_seqlens, output_cache=True, cache=conv_cache
        )
        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(
            hidden_states, delta,
            self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g,
        )

        r = self.r_proj(xr)
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()  # -exp(-0.5) * sigmoid(...)
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0 or v_first is None:
            v_first = v
        else:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        kk = F.normalize(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim), dim=-1, p=2.0)
        k = fused_k_rwkv7(k, a, self.k_a)

        # dealing with left-padding
        if attention_mask is not None:
            v = v * am

        r, w, k, a = map(
            lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim),
            (r, w, k, a),
        )
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        if self.training or seq_len >= 64:
            o, recurrent_state = chunk_rwkv7(
                r=r,
                w=w,
                k=k,
                v=v,
                a=-kk,
                b=kk * a,
                scale=1.,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_mul_recurrent_rwkv7(
                r=r,
                w=w,
                k=k,
                v=v,
                kk=kk,
                a=a,
                scale=1.,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=conv_state,
                layer_idx=self.layer_idx,
                offset=r.shape[1]
            )

        o = self.g_norm(rearrange(o, 'b t h d -> (b t) (h d)')).view(batch_size, seq_len, -1)
        o = gate_output_correction(o, r, k, self.r_k, v, g)
        o = self.o_proj(o)

        return o, v_first

    def state_size(self, sequence_length: int = 2048):
        return self.num_heads * self.head_dim * self.head_dim




class RWKV7Block(nn.Module):
    """Zoology block for vanilla RWKV-7 (RMSNorm + mixer + residual, no MLP)."""

    def __init__(self, config, layer_idx: int = None, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mixer = config.sequence_mixer.instantiate(
            d_model=config.d_model,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states, residual=None, v_first=None):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual)
        hidden_states, v_first = self.mixer(hidden_states, v_first=v_first)
        return hidden_states, residual, v_first
