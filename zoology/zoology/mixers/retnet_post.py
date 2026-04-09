# -*- coding: utf-8 -*-
# RetNet (Multi-Scale Retention) with PoST modifications for Zoology MQAR.
#
# Standard RetNet uses fixed exponential decay via rotary embeddings.
# RetNet PoST replaces this with PoST's learnable geometric spectral
# allocation + position-adaptive scaling, using the GLA chunk kernel
# (which accepts explicit per-head log-decay gates).

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.activations import ACT2FN
from fla.ops.gla import chunk_gla, fused_recurrent_gla

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


class RetNetPoST(nn.Module):
    """
    RetNet with PoST modifications for MQAR benchmarking.

    Standard RetNet uses fixed exponential decay (via rotary embeddings).
    RetNet PoST replaces this with:
      1. Cumulative softplus decay allocation (geometric ordering).
      2. Position-adaptive scaling on the log-decay gate.

    Uses the GLA chunk kernel with explicit decay gates instead of
    RetNet's rotary-based chunk_retention kernel.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        mode: str = 'chunk',
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        # PoST parameters
        train_length: int = 512,
        position_adaptive: bool = True,
        alpha_mode: str = 'analytical',
        **kwargs,
    ):
        super().__init__()

        self.mode = mode
        hidden_size = int(d_model)
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_short_conv = use_short_conv
        self.use_output_gate = use_output_gate
        self.layer_idx = layer_idx

        # PoST config
        self.position_adaptive = position_adaptive
        self.alpha_mode = alpha_mode
        self.log_t_ref = math.log(train_length)

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)

        assert self.key_dim % num_heads == 0
        assert self.value_dim % num_heads == 0

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # ---- Projections ----
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        # ---- PoST decay: geometric spectral allocation ----
        H = num_heads
        self.A_log_base = nn.Parameter(torch.tensor(0.0))
        self.A_log_delta = nn.Parameter(torch.zeros(max(H - 1, 1)))

        # Data-dependent gate scaling
        self.a_proj = nn.Linear(hidden_size, self.key_dim, bias=False)

        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.key_dim) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Alpha for position-adaptive scaling
        if self.alpha_mode == 'learnable' and H > 1:
            self.alpha_logits = nn.Parameter(torch.zeros(H - 1))
        elif H > 1:
            alpha = 1.0 - torch.arange(H, dtype=torch.float32) / (H - 1)
            self.register_buffer('alpha_fixed', alpha)
        else:
            self.register_buffer('alpha_fixed', torch.zeros(1))

        # ---- Output norm/gate ----
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if gate_fn == 'swish' and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(hidden_size=self.head_v_dim, eps=norm_eps)
            if use_output_gate:
                self.gate_fn = ACT2FN[gate_fn]

        # Initialize
        self._post_init()

    @torch.no_grad()
    def _post_init(self):
        H = self.num_heads
        target = torch.linspace(0.0, math.log(16.0), H)
        self.A_log_base.data = torch.tensor(target[0].item())
        if H > 1:
            gaps = target[1:] - target[:-1]
            self.A_log_delta.data = torch.log(torch.exp(gaps) - 1 + 1e-6)

    def get_A_log(self):
        if self.num_heads == 1:
            return self.A_log_base.unsqueeze(0)
        gaps = F.softplus(self.A_log_delta)
        zeros = torch.zeros(1, device=self.A_log_base.device, dtype=gaps.dtype)
        offsets = torch.cat([zeros, torch.cumsum(gaps, dim=0)])
        return self.A_log_base + offsets

    def get_alpha(self):
        H = self.num_heads
        if self.alpha_mode == 'learnable' and hasattr(self, 'alpha_logits'):
            gaps = F.softmax(self.alpha_logits, dim=0)
            cum = torch.cumsum(gaps, dim=0)
            ones = torch.ones(1, device=cum.device, dtype=cum.dtype)
            return torch.cat([ones, 1.0 - cum])

        if self.alpha_mode == 'analytical':
            A_log = self.get_A_log()
            cum = A_log - A_log[0]
            total = cum[-1]
            k = torch.arange(H, device=A_log.device, dtype=A_log.dtype)
            mean_gap = total / max(H - 1, 1)
            linear_taper = (H - 1 - k) / max(H - 1, 1)
            correction = (cum - k * mean_gap) / self.log_t_ref
            return (linear_taper + correction).clamp(min=0.0, max=1.0)

        return self.alpha_fixed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        mode = self.mode
        batch_size, seq_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -seq_len:] if attention_mask is not None else None
            position_ids = kwargs.get('position_ids', None)
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states), mask=conv_mask,
                                            cache=conv_state_q, output_final_state=use_cache,
                                            seq_idx=position_ids)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states), mask=conv_mask,
                                            cache=conv_state_k, output_final_state=use_cache,
                                            seq_idx=position_ids)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states), mask=conv_mask,
                                            cache=conv_state_v, output_final_state=use_cache,
                                            seq_idx=position_ids)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])

        q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_k_dim)
        k = rearrange(k, 'b t (h d) -> b t h d', d=self.head_k_dim)
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        # ---- PoST decay gate ----
        A_log = self.get_A_log()  # (H,)
        a = rearrange(self.a_proj(hidden_states), 'b t (h d) -> b t h d', d=self.head_k_dim)

        gk = -A_log.float().exp().unsqueeze(-1) * F.softplus(
            a.float() + self.dt_bias.view(self.num_heads, self.head_k_dim).float()
        )

        # Position-adaptive scaling
        if self.position_adaptive:
            alpha = self.get_alpha()
            positions = torch.arange(1, 1 + seq_len, device=gk.device, dtype=torch.float32)
            scale = positions.unsqueeze(-1) ** (-alpha.unsqueeze(0))  # (T, H)
            gk = gk * scale.unsqueeze(0).unsqueeze(-1)  # (B, T, H, D)

        # Use GLA chunk kernel with explicit decay gate
        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        if mode == 'chunk':
            o, recurrent_state = chunk_gla(
                q=q, k=k, v=v, g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q, k=k, v=v, gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1],
            )

        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, 'b t (h d) -> b t h d', d=self.head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, 'b t h d -> b t (h d)')
            else:
                o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')

        o = self.o_proj(o)
        return o

    def state_size(self, sequence_length: int = 2048):
        return self.num_heads * self.head_k_dim * self.head_v_dim
