# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for PoST Project — Position-Adaptive Decay via Geometric Spectral Allocation
#
# Based on the RWKV-Vibe/rwkv-fla RWKV-7 implementation.
# Only the decay initialization (w0 bias) and position-adaptive scaling are modified.
# All Triton kernels (chunk_rwkv7, fused_recurrent_rwkv7) remain unmodified.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from fla.layers.rwkv6 import LoRA
from fla.modules.token_shift import token_shift
from fla.ops.rwkv7 import chunk_rwkv7, fused_mul_recurrent_rwkv7
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7
from fla.ops.rwkv7.gate_output_correction import gate_output_correction

if TYPE_CHECKING:
    from fla.models.utils import Cache


# ==========================================
# 1. Configuration
# ==========================================

class RWKV7Config(PretrainedConfig):
    """Configuration for the PoST-RWKV-7 model."""
    model_type = "rwkv7_post"

    def __init__(
        self,
        d_model=64,
        n_layer=2,
        head_dim=64,
        decay_low_rank_dim=None,
        gate_low_rank_dim=None,
        a_low_rank_dim=None,
        v_low_rank_dim=None,
        hidden_ratio=4,
        dim_ffn=None,
        vocab_size=8192,
        pad_vocab_size_multiple=16,
        tie_embeddings=True,
        train_length=512,
        norm_eps=1e-5,
        use_post=True,                 # Enable PoST position-adaptive scaling
        **kwargs,
    ):
        self.d_model = d_model
        self.n_layer = n_layer
        self.head_dim = head_dim
        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.hidden_ratio = hidden_ratio
        if dim_ffn is None:
            dim_ffn = int(d_model * hidden_ratio)
            dim_ffn = 32 * ((dim_ffn + 31) // 32)  # round up to multiple of 32
        self.dim_ffn = dim_ffn
        self.vocab_size = vocab_size
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.tie_embeddings = tie_embeddings
        self.train_length = train_length
        self.norm_eps = norm_eps
        self.use_post = use_post

        kwargs["tie_word_embeddings"] = tie_embeddings
        super().__init__(**kwargs)

        # Derived
        self.num_heads = d_model // head_dim
        if d_model % head_dim != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by head_dim ({head_dim})")

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)


# ==========================================
# 2. PoST-RWKV-7 Attention Layer
# ==========================================

class RWKV7PoSTAttention(nn.Module):
    """
    RWKV-7 attention with PoST geometric decay allocation.

    Built on the official RWKV-Vibe/rwkv-fla implementation with two targeted
    modifications:

    1. ``w0`` ordering via PoST map — the per-channel base decay bias is
       initialized and trained via the cumulative Softplus parameterization
       ``w0_k = θ + Σ_{j<k} softplus(δ_j)``, guaranteeing geometric spectral
       ordering throughout training. This is the sole PoST modification.

    Everything else — runtime decay computation, LoRA modulation, fused ops,
    token-shift, GroupNorm, chunk_rwkv7/fused_mul_recurrent_rwkv7 kernel —
    is identical to the official implementation.
    """

    def __init__(
        self,
        config: RWKV7Config,
        layer_idx: int = 0,
    ):
        super().__init__()

        hidden_size = config.d_model
        head_dim = config.head_dim
        num_heads = config.num_heads
        n_layer = config.n_layer

        self.hidden_size = hidden_size
        self.key_dim = hidden_size
        self.value_dim = hidden_size
        self.head_dim = head_dim
        self.head_v_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.num_hidden_layers = n_layer
        self.use_post = config.use_post

        self.log_t_ref = math.log(config.train_length)

        # ---- LoRA rank calculation (follows official RWKV-7 defaults) ----
        factor = head_dim / 64
        decay_low_rank_dim = config.decay_low_rank_dim or max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
        gate_low_rank_dim = config.gate_low_rank_dim or max(32, int(round((5 * (hidden_size**0.5)) / 32) * 32))
        a_low_rank_dim = config.a_low_rank_dim or max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
        v_low_rank_dim = config.v_low_rank_dim or max(32, int(round((1.7 * (hidden_size**0.5)) * factor / 32) * 32))

        # ---- Standard RWKV-7 parameters ----
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(num_heads, head_dim))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # PoST mode:   bias=True, initialized to strictly zigzag*2.5.
        #              w0 provides the global ordered timescale curve, and the 
        #              zigzag bias provides necessary intra-head variation.
        # Vanilla mode: bias=True, initialised to www + 0.5 + zigzag*2.5.
        self.w_lora = LoRA(hidden_size, self.key_dim,
                           low_rank_dim=decay_low_rank_dim, activation='tanh',
                           bias=True)
        if layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim,
                               low_rank_dim=v_low_rank_dim, activation=None)
        self.a_lora = LoRA(hidden_size, self.key_dim,
                           low_rank_dim=a_low_rank_dim, activation=None)
        self.g_lora = LoRA(hidden_size, self.value_dim,
                           low_rank_dim=gate_low_rank_dim, activation='sigmoid', bias=False)

        self.g_norm = nn.GroupNorm(
            num_groups=num_heads,
            num_channels=self.value_dim,
            eps=head_dim * config.norm_eps,
            affine=True,
        )

        # ---- PoST decay parameterization (global per-channel) ----
        C = self.key_dim
        # PoST map: w0_k = theta - sum_{j<k} softplus(delta_j)  [strictly decreasing]
        if self.use_post:
            self.w0_base = nn.Parameter(torch.tensor(-6.0))
            self.w0_base._lr_scale = 2.0   # official RWKV-7: att.w0 gets 2× LR
            self.w0_delta = nn.Parameter(torch.zeros(C - 1))
            self.w0_delta._lr_scale = 2.0  # official RWKV-7: att.w0 gets 2× LR


        # ---- Initialize ----
        self._post_init(config)

    @torch.no_grad()
    def _post_init(self, config: RWKV7Config):
        """Initialize parameters with PoST map for decay, official patterns for rest."""
        hidden_size = self.hidden_size
        n_layer = self.num_hidden_layers
        layer_idx = self.layer_idx

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

        self.x_r.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_r.dtype)
        self.x_w.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_w.dtype)
        self.x_k.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).to(self.x_k.dtype)
        self.x_v.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).to(self.x_v.dtype)
        self.x_a.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_a.dtype)
        self.x_g.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_g.dtype)

        nn.init.constant_(self.k_a, 1.02)
        nn.init.constant_(self.r_k, -0.04)
        self.k_k.data.copy_((torch.zeros(hidden_size) + 0.71 - linear * 0.1).to(self.k_k.dtype))

        # w_lora bias init
        if self.use_post:
            self.w_lora.set_bias_value(zigzag * 2.5)
        else:
            self.w_lora.set_bias_value(www + 0.5 + zigzag * 2.5)

        self.a_lora.set_bias_value(-0.19 + zigzag * 0.3 + linear * 0.4)

        if layer_idx != 0:
            self.v_lora._initialize_weights(self.v_lora)
            self.v_lora.set_bias_value(0.73 - linear * 0.4)

        if n_layer > 0:
            self.g_norm.weight.data[:] = ((layer_idx + 1) / n_layer) ** 0.7

        self._orthogonal_init(self.r_proj.weight)
        self._orthogonal_init(self.k_proj.weight, gain=0.1)
        self._orthogonal_init(self.v_proj.weight)
        self.o_proj.weight.data.zero_()

        if self.use_post:
            # PoST map init for additive logit-space scaling.
            #
            # Taper convention (matches Mamba-2):
            #   k=0 is the SLOW channel (α=1), k=C-1 is the FAST channel (α=0).
            # Practical timescale: τ_k(t) = t^{α_k} / (0.5 + |log σ(w0_k)|)
            #
            # w0 ∈ [6.5, 0.5] decreasing (no separate LoRA bias):
            #
            # Slow channel (k=0, α=1, w0=6.5): p_0 = log σ(6.5) ≈ -0.0015
            #   τ_0(t) = t/(0.5 + 0.0015) ≈ 2t  → at t=T_train: τ ≈ 2T   ← very slow ✓
            #
            # Fast channel (k=C-1, α=0, w0=0.5): p_N = log σ(0.5) ≈ -0.476
            #   τ_N = 1/(0.5 + 0.476) ≈ 1.02  (constant ≈ 1 step) ← fast ✓
            #
            # Timescale range at t=T_train: [1, 2T].
            # Timescale range at t=T_train: [1, T_train].
            C = self.key_dim
            T_train = config.train_length
            x_slow = 1.0 / (0.6065306597126334 * T_train)
            w0_target_slow = -math.log((1.0 / x_slow) - 1.0)
            
            w0_target_fast = 0.5    # k=C-1: τ ≈ 2.65 (fast, α=0) matches vanilla upper bound
            target_w0 = torch.linspace(w0_target_slow, w0_target_fast, C)  # increasing
            self.w0_base.data = target_w0[0].clone()  # smallest
            real_gaps = target_w0[1:] - target_w0[:-1]  # positive (since increasing)
            self.w0_delta.data = torch.log(torch.exp(real_gaps) - 1 + 1e-6)

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

    def get_w0(self):
        """Compute base decay bias via PoST map (cumulative Softplus, increasing)."""
        gaps = F.softplus(self.w0_delta)
        zeros = torch.zeros(1, device=self.w0_base.device, dtype=gaps.dtype)
        offsets = torch.cat([zeros, torch.cumsum(gaps, dim=0)])
        return self.w0_base + offsets  # (C,), strictly increasing

    def get_alpha(self):
        """Per-channel taper exponents α_k for additive logit-space position scaling.

        Convention matches Mamba-2: k=0 is SLOW (α=1), k=C-1 is FAST (α=0).

        Uses p_k = logsigmoid(w0_k) as the spectral proxy.
        Because w0 is INCREASING, p_k is also INCREASING (e.g. from -5 to -0.47).
        This matches the geometry of Mamba-2's -A_log.

        Taper direction (DECREASING in k — matches Mamba-2):
          k=0 (smallest w0):          α=1 → τ ≈ t  [slow, grows with t]
          k=C-1 (largest w0):         α=0 → τ ≈ 1   [fast, constant]

        The correction term compensates for non-uniform |p|-gaps (logsigmoid is non-linear).
        """
        C = self.key_dim
        w0 = self.get_w0()  # (C,), increasing
        # p = logsigmoid(w0): INCREASING (analogous to Mamba-2's A_log)
        p = F.logsigmoid(w0)       # (C,), negative, increasing
        cum = p - p[0]             # cum[0]=0, positive, increasing
        k = torch.arange(C, device=w0.device, dtype=w0.dtype)
        mean_gap = cum[-1] / max(C - 1, 1)
        # DECREASING taper: k=0 → α=1 (slow), k=C-1 → α=0 (fast)
        linear_taper = (C - 1 - k) / max(C - 1, 1)
        correction = (cum - k * mean_gap) / self.log_t_ref
        return (linear_taper + correction).clamp(min=0.0, max=1.0)  # (C,)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        v_first: torch.Tensor = None,
        position_offset: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional[object], torch.Tensor]:
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

        delta, conv_state = token_shift(
            hidden_states, cu_seqlens, output_cache=True, cache=conv_cache
        )
        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(
            hidden_states, delta,
            self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g,
        )

        r = self.r_proj(xr)
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        # Decay computation
        if self.use_post:
            # PoST: additive logit-space scaling.
            #
            #   logit_eff(t, k) = w0_k + LoRA(x_t)_k − α_k · log(t)
            #   w_t,k = −0.6065 · σ(logit_eff(t, k))
            #
            # w0 provides the base operating point (no separate bias).
            # LoRA provides content-dependent modulation (bias=False).
            # Position scaling subtracts α_k·log(t) from the total logit.
            #
            # Timescale: for slow channels (logit≪-1),
            #   τ_k(t) ≈ t^α_k / (0.6065·exp(w0_k))  ∝  t^α_k
            w0    = self.get_w0().float()    # (C,)
            alpha = self.get_alpha().float() # (C,)

            positions = torch.arange(
                1 + position_offset,
                1 + position_offset + seq_len,
                device=hidden_states.device, dtype=torch.float32,
            )                                                    # (T,)
            log_t = torch.log(positions)                         # (T,)

            # LoRA output: content-dependent modulation only (no bias)
            lora_content = self.w_lora(xw).float()               # (B, T, C)

            # Effective logit = PoST base + content + position taper
            logit_eff = (w0[None, None, :]
                         + lora_content
                         - alpha[None, None, :] * log_t[None, :, None])  # (B, T, C)
            w = -0.6065306597126334 * torch.sigmoid(logit_eff)   # (B, T, C)
        else:
            # Vanilla RWKV-7 mode: LoRA with bias (FLA's original)
            w = -0.6065306597126334 * self.w_lora(xw).sigmoid()

        # ============================================================
        # Standard RWKV-7 from here — completely unchanged
        # ============================================================
        if self.layer_idx == 0 or v_first is None:
            v_first = v
        else:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        kk = F.normalize(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim), dim=-1, p=2.0)
        k = fused_k_rwkv7(k, a, self.k_a)

        if attention_mask is not None:
            v = v * am

        r, w, k, a = map(
            lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim),
            (r, w, k, a),
        )
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        if self.training or seq_len >= 64:
            o, recurrent_state = chunk_rwkv7(
                r=r, w=w, k=k, v=v,
                a=-kk, b=kk * a,
                scale=1.,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_mul_recurrent_rwkv7(
                r=r, w=w, k=k, v=v,
                kk=kk, a=a,
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
                offset=r.shape[1],
            )

        o = self.g_norm(rearrange(o, 'b t h d -> (b t) (h d)')).view(batch_size, seq_len, -1)
        o = gate_output_correction(o, r, k, self.r_k, v, g)
        o = self.o_proj(o)

        return o, None, past_key_values, v_first


# ==========================================
# 3. RWKV-7 Channel Mix (FFN)
# ==========================================

class RWKV7CMix(nn.Module):
    """
    RWKV-7 Channel Mix (FFN) following FLA's RWKV7FeedForward.

    Architecture: token-shift → key projection → sqrelu → value projection
    Default dim_ffn = d_model * hidden_ratio (rounded to 32).
    """

    def __init__(self, config: RWKV7Config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.d_model
        self.dim_ffn = config.dim_ffn
        self.layer_idx = layer_idx
        self.num_hidden_layers = config.n_layer

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(config.d_model))

        self.key = nn.Linear(config.d_model, self.dim_ffn, bias=False)
        self.value = nn.Linear(self.dim_ffn, config.d_model, bias=False)

        self._initialize_weights()

    @torch.no_grad()
    def _initialize_weights(self):
        """Official RWKV-7 CMix initialization (from FLA's RWKV7FeedForward)."""
        ratio_1_to_almost0 = 1.0 - (self.layer_idx / self.num_hidden_layers)
        ddd = torch.ones(1, 1, self.hidden_size)
        for i in range(self.hidden_size):
            ddd[0, 0, i] = i / self.hidden_size
        self.x_k.data = (1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4)).squeeze()

        # Key: orthogonal init; Value: zero init
        original_dtype = self.key.weight.dtype
        self.key.weight.data = nn.init.orthogonal_(
            self.key.weight.data.to(torch.float32)
        ).to(original_dtype)
        self.value.weight.data.zero_()

        # Protect from external reinit
        for p in self.parameters():
            p._no_reinit = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


# ==========================================
# 4. Norm (use nn.LayerNorm to match FLA)
# ==========================================


# ==========================================
# 5. Full CausalLM Model
# ==========================================

class RWKV7PoSTForCausalLM(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"head.weight": "embedding.weight"}
    """Complete RWKV-7 language model with PoST decay parameterization."""
    config_class = RWKV7Config

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
        }

    def __init__(self, config: RWKV7Config):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Pre-norm for layer 0 (matches FLA's norm_first)
        self.pre_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                mixer=RWKV7PoSTAttention(config, layer_idx=i),
                norm=nn.LayerNorm(config.d_model, eps=config.norm_eps),
                ffn=RWKV7CMix(config, layer_idx=i),
                ffn_norm=nn.LayerNorm(config.d_model, eps=config.norm_eps),
            ))
            for i in range(config.n_layer)
        ])

        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.head.weight = self.embedding.weight

        # Trigger _init_weights on all modules (embedding, lm_head, etc.)
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    @torch.no_grad()
    def _init_weights(self, module):
        """Weight initialization following FLA's RWKV7PreTrainedModel."""
        # Skip modules already custom-initialized by RWKV7 attention/FFN
        if getattr(module, '_no_reinit', False):
            return
        if getattr(module, '_is_hf_initialized', False):
            return

        if isinstance(module, nn.Embedding):
            # FLA: tiny uniform init for embeddings
            nn.init.uniform_(module.weight, a=-1e-4, b=1e-4)
        elif isinstance(module, nn.Linear) and hasattr(self, 'head') and module is self.head:
            # FLA: orthogonal init for lm_head with scaled gain
            # Skip if tied to embedding (shared weight already gets tiny uniform init)
            if not self.config.tie_embeddings:
                if self.config.vocab_size > self.config.d_model:
                    scale = 0.5 * math.sqrt(self.config.vocab_size / self.config.d_model)
                else:
                    scale = 0.5
                original_dtype = module.weight.dtype
                module.weight.data = nn.init.orthogonal_(
                    module.weight.data.to(torch.float32), gain=scale
                ).to(original_dtype)

    def forward(self, input_ids, labels=None, attention_mask=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        x = self.embedding(input_ids)
        v_first = None

        # Apply pre-norm at layer 0 (FLA's norm_first)
        x = self.pre_norm(x)

        for layer in self.layers:
            residual = x
            x_norm = layer['norm'](x)
            out, _, _, v_first = layer['mixer'](x_norm, v_first=v_first)
            x = out + residual
            # Channel Mix (FFN)
            residual = x
            x = layer['ffn'](layer['ffn_norm'](x)) + residual

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
