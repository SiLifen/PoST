# -*- coding: utf-8 -*-
# GLA (Gated Linear Attention) with PoST modifications for Zoology MQAR.
#
# Extends the standard GLA by replacing the data-dependent gate
# with PoST's geometric spectral allocation + position-adaptive scaling.
# All projections and kernels are identical to baseline GLA.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.activations import ACT2FN
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


class GLAPoST(nn.Module):
    """
    Gated Linear Attention with PoST modifications for MQAR benchmarking.

    Two changes from the standard GLA:
      1. Gate (gk) uses cumulative softplus (geometric ordering) for spectral allocation.
      2. Position-adaptive scaling on the log-decay gate.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        d_model: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        feature_map: Optional[str] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        fuse_norm: bool = True,
        layer_idx: int = None,
        # PoST parameters
        train_length: int = 512,
        position_adaptive: bool = True,
        alpha_mode: str = 'analytical',
        **kwargs,
    ) -> "GLAPoST":
        super().__init__()

        self.mode = mode

        hidden_size = int(d_model)
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.feature_map_fn = ACT2FN[feature_map] if feature_map is not None else None

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_output_gate = use_output_gate

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.layer_idx = layer_idx

        # PoST config
        self.position_adaptive = position_adaptive
        self.alpha_mode = alpha_mode
        self.log_t_ref = math.log(train_length)

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not supported mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # ---- Projections (identical to standard GLA) ----
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim_per_group, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim_per_group, conv_size, activation='silu')

        # ---- PoST decay: geometric spectral allocation ----
        # Replaces GLA's gk_proj with structured spectral decay
        H = num_heads
        self.A_log_base = nn.Parameter(torch.tensor(0.0))
        self.A_log_delta = nn.Parameter(torch.zeros(max(H - 1, 1)))

        # Data-dependent gate scaling (like GLA's gk_proj but PoST-structured)
        self.a_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)

        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.key_dim_per_group) * (math.log(dt_max) - math.log(dt_min))
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

        if gate_fn == 'swish' and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
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
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

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
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states),
                                            mask=conv_mask, cache=conv_state_q,
                                            output_final_state=use_cache, seq_idx=position_ids)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states),
                                            mask=conv_mask, cache=conv_state_k,
                                            output_final_state=use_cache, seq_idx=position_ids)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states),
                                            mask=conv_mask, cache=conv_state_v,
                                            output_final_state=use_cache, seq_idx=position_ids)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        if self.feature_map_fn is not None:
            q, k = map(self.feature_map_fn, (q, k))

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])

        q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_k_dim)
        if self.num_kv_groups > 1:
            k = repeat(k, 'b t (h d) -> b t (h g) d', g=self.num_kv_groups, d=self.head_k_dim)
            v = repeat(v, 'b t (h d) -> b t (h g) d', g=self.num_kv_groups, d=self.head_v_dim)
        else:
            k = rearrange(k, 'b t (h d) -> b t h d', d=self.head_k_dim)
            v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        # ---- PoST decay gate (replaces GLA's gk_proj) ----
        A_log = self.get_A_log()  # (H,)
        # Data-dependent component: project hidden_states and apply softplus
        a = self.a_proj(hidden_states)  # (B, T, key_dim_per_group)
        if self.num_kv_groups > 1:
            a = repeat(a, 'b t (h d) -> b t (h g) d', g=self.num_kv_groups, d=self.head_k_dim)
        else:
            a = rearrange(a, 'b t (h d) -> b t h d', d=self.head_k_dim)

        # gk = -exp(A_log_h) * softplus(a + dt_bias), per head per key_dim
        # A_log is (H,), dt_bias is (key_dim_per_group,), a is (B, T, H, head_k_dim)
        gk = -A_log.float().exp().unsqueeze(-1) * F.softplus(
            a.float() + self.dt_bias.view(self.num_kv_heads, self.head_k_dim).float()
        )

        # Position-adaptive scaling
        if self.position_adaptive:
            alpha = self.get_alpha()  # (H,)
            positions = torch.arange(1, 1 + seq_len, device=gk.device, dtype=torch.float32)
            # scale: (T, H) -> broadcast to (B, T, H, D)
            scale = positions.unsqueeze(-1) ** (-alpha.unsqueeze(0))  # (T, H)
            gk = gk * scale.unsqueeze(0).unsqueeze(-1)  # (B, T, H, D)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q, k=k, v=v, gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(
                q=q, k=k, v=v, g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                head_first=False,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(
                q=q, k=k, v=v, g=gk,
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

    def state_size(self, **kwargs) -> int:
        return self.key_dim * self.head_v_dim
