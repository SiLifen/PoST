# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for PoST Project — Position-Adaptive Decay via Geometric Spectral Allocation
#
# Based on the Gated DeltaNet implementation (arXiv:2412.06464).
# Only the decay initialization (A_log, dt_bias) and position-adaptive scaling are modified.
# All Triton kernels (chunk_gated_delta_rule) remain unmodified.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

if TYPE_CHECKING:
    from fla.models.utils import Cache


# ==========================================
# 1. Configuration
# ==========================================

class GatedDeltaNetConfig(PretrainedConfig):
    """Configuration for the PoST-GatedDeltaNet model."""
    model_type = "post_gated_deltanet"

    def __init__(
        self,
        d_model=768,
        n_layer=24,
        num_heads=6,
        expand_v=2,
        use_gate=True,
        use_short_conv=True,
        conv_size=4,
        vocab_size=8192,
        pad_vocab_size_multiple=16,
        tie_embeddings=True,
        train_length=512,
        norm_eps=1e-5,
        use_post=True,                  # Enable PoST position-adaptive scaling
        **kwargs,
    ):
        self.d_model = d_model
        self.n_layer = n_layer
        self.num_heads = num_heads
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.vocab_size = vocab_size
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.tie_embeddings = tie_embeddings
        self.train_length = train_length
        self.norm_eps = norm_eps
        self.use_post = use_post

        kwargs["tie_word_embeddings"] = tie_embeddings
        super().__init__(**kwargs)

        # Derived
        self.head_dim = d_model // num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)


# ==========================================
# 2. PoST-GatedDeltaNet Layer
# ==========================================

class GDNPoST(GatedDeltaNet):
    """
    Gated DeltaNet with optional PoST geometric decay allocation.

    Inherits from fla's GatedDeltaNet and optionally overrides:

    1. ``A_log`` initialization — replaced the uniform random [0, 16] range
       with the PoST map: cumulative Softplus parameterization guaranteeing
       geometric spectral ordering throughout training.

    2. Position-adaptive scaling — after computing the log-decay
       ``g = -exp(A_log) * softplus(a(x) + dt_bias)``, the effective log-decay
       is scaled by ``1/t^{alpha_h}`` per head before passing to the kernel.

    When use_post=False, inherits all behavior from fla's GatedDeltaNet
    unchanged — same parameters, same forward path.
    """

    def __init__(
        self,
        config: GatedDeltaNetConfig,
        layer_idx: int = 0,
    ):
        hidden_size = config.d_model
        num_heads = config.num_heads
        head_dim = config.head_dim
        expand_v = config.expand_v

        # Initialize parent (fla's GatedDeltaNet)
        super().__init__(
            hidden_size=hidden_size,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            use_gate=config.use_gate,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            layer_idx=layer_idx,
            norm_eps=config.norm_eps,
        )

        self.use_post = config.use_post

        # --- PoST-specific overrides (skip when use_post=False to keep vanilla GDN) ---
        if self.use_post:
            self.log_t_ref = math.log(config.train_length)

            # Override A_log: PoST map with cumulative Softplus
            H = num_heads
            del self.A_log

            self.A_log_base = nn.Parameter(torch.tensor(0.0))
            self.A_log_delta = nn.Parameter(torch.zeros(H - 1))

            # Initialize with geometric spacing
            self._post_init_decay(H, config.train_length)

    @torch.no_grad()
    def _post_init_decay(self, H: int, train_length: int):
        """Initialize decay parameters with PoST geometric allocation."""
        # PoST map: geometric spacing across heads
        # A_log range spanning [0, log(train_length)] for full spectral coverage
        device = self.A_log_base.device
        target_A_log = torch.linspace(0.0, math.log(train_length), H, device=device)
        self.A_log_base.data = target_A_log[0]
        real_gaps = target_A_log[1:] - target_A_log[:-1]
        self.A_log_delta.data = torch.log(torch.exp(real_gaps) - 1 + 1e-6)

    def get_A_log(self):
        """Compute A_log via PoST map (cumulative Softplus) or return parent's A_log."""
        if not self.use_post:
            return self.A_log
        gaps = F.softplus(self.A_log_delta)
        zeros = torch.zeros(1, device=self.A_log_base.device, dtype=gaps.dtype)
        offsets = torch.cat([zeros, torch.cumsum(gaps, dim=0)])
        return self.A_log_base + offsets  # (H,), strictly ordered

    def get_alpha(self):
        """Get position-adaptive taper exponents (analytical mode)."""
        A_log = self.get_A_log()
        H = A_log.shape[0]
        cum = A_log - A_log[0]
        total = cum[-1]
        k = torch.arange(H, device=A_log.device, dtype=A_log.dtype)
        mean_gap = total / max(H - 1, 1)
        linear_taper = (H - 1 - k) / max(H - 1, 1)
        correction = (cum - k * mean_gap) / self.log_t_ref
        return (linear_taper + correction).clamp(min=0.0, max=1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_offset: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional[object]]:
        """
        Forward pass. When use_post=False, delegates to parent (fla's GatedDeltaNet)
        for identical behavior. When use_post=True, uses PoST position-adaptive scaling.
        """
        # --- Vanilla GDN: delegate to parent's forward ---
        if not self.use_post:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

        # --- PoST: forward with position-adaptive Δ-scaling ---
        batch_size, seq_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # Short convolutions
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -seq_len:] if attention_mask is not None else None
            cu_seqlens = kwargs.get('cu_seqlens', None)
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states),
                                            cache=conv_state_q, output_final_state=use_cache,
                                            cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states),
                                            cache=conv_state_k, output_final_state=use_cache,
                                            cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states),
                                            cache=conv_state_v, output_final_state=use_cache,
                                            cu_seqlens=cu_seqlens)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        beta = self.b_proj(hidden_states).sigmoid()

        # ============================================================
        # Decay computation — PoST modifies GDN here
        # ============================================================
        A_log = self.get_A_log()  # (H,)
        g = -A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)
        # g shape: (B, T, H) — log-decay per head

        # Position-adaptive scaling
        alpha = self.get_alpha()  # (H,)
        positions = torch.arange(
            1 + position_offset,
            1 + position_offset + seq_len,
            device=g.device, dtype=torch.float32,
        )
        scale = positions.unsqueeze(-1) ** (-alpha.unsqueeze(0))  # (T, H)
        g = g * scale.unsqueeze(0)  # (B, T, H)

        # ============================================================
        # Standard GDN from here — completely unchanged
        # ============================================================
        if attention_mask is not None:
            am = attention_mask[:, -seq_len:]
            beta = beta.mul(am.unsqueeze(-1))
            g = g.mul(am.unsqueeze(-1))

        q, k, v, beta, g = [x.to(torch.bfloat16) for x in (q, k, v, beta, g)]

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        if self.training or seq_len >= 64:
            o, recurrent_state = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1],
            )

        if self.use_gate:
            g_out = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)').to(hidden_states.dtype)
        o = self.o_proj(o)

        return o, None, past_key_values


# ==========================================
# 3. RMSNorm
# ==========================================

class RMSNormSimple(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ==========================================
# 4. Full CausalLM Model
# ==========================================

class GDNPoSTForCausalLM(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"head.weight": "embedding.weight"}
    """Complete Gated DeltaNet language model with optional PoST decay parameterization."""
    config_class = GatedDeltaNetConfig

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
        }

    def __init__(self, config: GatedDeltaNetConfig):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                mixer=GDNPoST(config, layer_idx=i),
                norm=RMSNormSimple(config.d_model),
            ))
            for i in range(config.n_layer)
        ])

        self.norm_f = RMSNormSimple(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.head.weight = self.embedding.weight

        # Initialize weights
        self.post_init()

    def _init_weights(self, module):
        """Weight initialization matching FLA's GatedDeltaNetPreTrainedModel._init_weights.

        NOTE: A_log and dt_bias are NOT re-initialized here.
        They are already properly initialized in GDNPoST.__init__.
        Re-initializing them here would break from_pretrained(), which calls
        _init_weights AFTER loading checkpoint weights — overwriting trained
        values with random ones.
        """
        initializer_range = getattr(self.config, 'initializer_range', 0.02)

        # --- GatedDeltaNet layer: only set metadata attributes ---
        if isinstance(module, GDNPoST):
            if hasattr(module, 'A_log'):
                module.A_log._no_weight_decay = True
            if hasattr(module, 'A_log_base'):
                module.A_log_base._no_weight_decay = True
            if hasattr(module, 'A_log_delta'):
                module.A_log_delta._no_weight_decay = True
            if hasattr(module, 'dt_bias'):
                module.dt_bias._no_weight_decay = True

        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def forward(self, input_ids, labels=None, attention_mask=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        x = self.embedding(input_ids)

        for layer in self.layers:
            residual = x
            x_norm = layer['norm'](x)
            out, _, _ = layer['mixer'](x_norm)
            x = out + residual

        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
