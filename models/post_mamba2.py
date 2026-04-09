# Copyright (c) 2024, Tri Dao, Albert Gu.
# Modified for PoST Project — Position-Adaptive SSM via Δ-Scaling
#
# Based on fla's Mamba2 implementation (fla.layers.mamba2).
# PoST extends the standard Mamba-2 layer by:
#   1. Replacing free A_log with constrained monotonic parameterization
#   2. Using deterministic dt_bias initialization
#   3. Adding position-adaptive Δ-scaling in the forward pass

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

# Import fla's Mamba2 layer as base
from fla.layers.mamba2 import Mamba2
from fla.models.mamba2.configuration_mamba2 import Mamba2Config as FlaMamba2Config
from fla.modules.layernorm import RMSNorm

# Import kernels (same ones fla uses)
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from post_kernels.ssd_combined import mamba_chunk_scan_combined

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


# ==========================================
# 1. Configuration — extends fla's Mamba2Config
# ==========================================

class Mamba2PoSTConfig(FlaMamba2Config):
    """Mamba-2 config with PoST-specific fields."""
    model_type = "mamba2_post"

    def __init__(
        self,
        train_length=2048,
        use_post=True,
        **kwargs,
    ):
        # PoST-specific
        self.train_length = train_length
        self.use_post = use_post

        super().__init__(**kwargs)


# ==========================================
# 2. PoST Mamba-2 Layer — extends fla's Mamba2
# ==========================================

class Mamba2PoST(Mamba2):
    """
    Mamba-2 layer with PoST position-adaptive discretization.

    Inherits from fla's Mamba2 and overrides:
    - A parameterization: A_log_base + cumsum(softplus(deltas)) for monotonic ordering
    - forward: non-fused path with position-adaptive A-scaling
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int = 64,
        hidden_size: int = 2048,
        state_size: int = 128,
        expand: int = 2,
        n_groups: int = 1,
        conv_kernel: int = 4,
        use_conv_bias: bool = False,
        hidden_act: str = "silu",
        rms_norm: bool = True,
        chunk_size: int = 256,
        time_step_rank: float = 256,
        time_step_limit: tuple = (0.0, float("inf")),
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        use_bias: bool = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        # PoST-specific
        train_length: int = 2048,
        use_post: bool = True,
        **kwargs,
    ):
        # Initialize parent (fla's Mamba2) — gives us standard A_log, dt_bias, etc.
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            state_size=state_size,
            expand=expand,
            n_groups=n_groups,
            conv_kernel=conv_kernel,
            use_conv_bias=use_conv_bias,
            hidden_act=hidden_act,
            rms_norm=rms_norm,
            chunk_size=chunk_size,
            time_step_rank=time_step_rank,
            time_step_limit=time_step_limit,
            time_step_min=time_step_min,
            time_step_max=time_step_max,
            use_bias=use_bias,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
        )

        self.use_post = use_post
        self.train_length = train_length

        # --- PoST-specific overrides (skip when use_post=False to keep vanilla Mamba2) ---
        if use_post:
            # Shared discretization scale: links dt_bias and A initialization
            # so that initial timescales span [1, train_length] exactly.
            base_dt = 0.05

            # Freeze dt_bias: uniform value, not learnable.
            # A is the sole spectrum controller; dt_proj(x) provides per-token modulation.
            dt_val = torch.full((self.num_heads,), base_dt)
            inv_dt = dt_val + torch.log(-torch.expm1(-dt_val))
            self.dt_bias = nn.Parameter(inv_dt, requires_grad=False)

            # Override A: PoST geometric spacing via base + cumulative softplus
            # Delete parent's A_log
            del self.A_log

            t_param_min = 1.0 * base_dt
            t_param_max = train_length * base_dt

            log_T = torch.linspace(
                math.log(t_param_min),
                math.log(t_param_max),
                self.num_heads,
            )
            target_A_log = -torch.flip(log_T, dims=[0])

            self.A_log_base = nn.Parameter(target_A_log[0], requires_grad=True)
            real_deltas = target_A_log[1:] - target_A_log[:-1]
            init_deltas_param = torch.log(torch.exp(real_deltas) - 1 + 1e-6)
            self.A_log_deltas = nn.Parameter(init_deltas_param, requires_grad=True)

            # For analytical alpha computation
            self.log_t_ref = math.log(train_length)

    def get_A_log(self):
        """Get A_log values with monotonic geometric spacing (PoST mode only)."""
        if not self.use_post:
            return self.A_log
        deltas = F.softplus(self.A_log_deltas)
        zeros = torch.zeros(1, device=self.A_log_base.device, dtype=deltas.dtype)
        offsets = torch.cat([zeros, torch.cumsum(deltas, dim=0)])
        return self.A_log_base + offsets

    def get_alpha(self):
        """Get normalization exponents α_k via analytical mode (Theorem 4.5).

        α_k = (N-k)/(N-1) + (cum_k - k·Ḡ) / log(T_train)
        Co-adapts with A during training, no extra parameters.
        """
        A_log = self.get_A_log()
        cum = A_log - A_log[0]
        total = cum[-1]
        N = self.num_heads
        k = torch.arange(N, device=A_log.device, dtype=A_log.dtype)
        mean_gap = total / (N - 1)
        linear_taper = (N - 1 - k) / (N - 1)
        correction = (cum - k * mean_gap) / self.log_t_ref
        alpha = linear_taper + correction
        return alpha.clamp(min=0.0, max=1.0)

    def forward(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        position_offset=0,
    ):
        """
        Forward pass. When use_post=False, delegates to parent (fla's Mamba2)
        for identical behavior including fused kernels. When use_post=True,
        uses a non-fused path with position-adaptive Δ-scaling.
        """
        # --- Vanilla Mamba-2: delegate to parent's forward (fused path) ---
        if not self.use_post:
            return super().forward(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

        # --- PoST: non-fused path with position-adaptive Δ-scaling ---
        batch_size, seq_len, _ = hidden_states.shape

        A = -torch.exp(self.get_A_log().float())  # (num_heads,)
        dt_limit_kwargs = (
            {} if self.time_step_limit == (0.0, float("inf"))
            else {"dt_limit": self.time_step_limit}
        )

        # 1. Input projection
        projected_states = self.in_proj(hidden_states)

        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2

        _, _, gate, hidden_states_B_C, dt_raw = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )

        # 2. Causal conv1d
        hidden_states_B_C = rearrange(
            causal_conv1d_fn(
                rearrange(hidden_states_B_C, "b s d -> b d s").contiguous(),
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
            ),
            "b d s -> b s d",
        )

        # 3. Split into x, B, C
        groups_time_state_size = self.n_groups * self.ssm_state_size
        x, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, groups_time_state_size, groups_time_state_size],
            dim=-1,
        )

        # 4. Compute dt with softplus
        dt = F.softplus(dt_raw + self.dt_bias)  # (batch, seq_len, num_heads)

        # 5. ★ PoST position-adaptive A-scaling ★
        #
        # Theory (Def 4.5): scale log w = A·Δ by 1/t^α → only affects decay.
        # We scale A directly via A_scale, so:
        #   Decay:  exp(A · s · Δ)  where s = t^(-α)     ✓ (theory)
        #   Input:  Δ · B · x                              ✓ (unchanged)
        #   D skip: D · x                                   ✓ (unchanged)
        # No x compensation or manual D skip needed.
        #
        positions = torch.arange(
            1 + position_offset,
            1 + position_offset + seq_len,
            device=dt.device,
            dtype=torch.float32,
        )  # (seq_len,)
        alpha = self.get_alpha()  # (num_heads,)
        A_scale = positions.unsqueeze(-1) ** (-alpha.unsqueeze(0))  # (seq_len, num_heads)
        A_scale = A_scale.unsqueeze(0).expand(batch_size, -1, -1).contiguous()  # (batch, seq_len, num_heads)

        # 6. SSD scan with A_scale
        x_bhp = rearrange(x, "b l (h p) -> b l h p", h=self.num_heads)
        scan_output = mamba_chunk_scan_combined(
            x_bhp,
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.n_groups),
            rearrange(C, "b l (g n) -> b l g n", g=self.n_groups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            dt_softplus=False,
            A_scale=A_scale,
            **dt_limit_kwargs,
        )

        # 7. Gated RMSNorm + output projection
        scan_output = rearrange(scan_output, "b s h p -> b s (h p)")
        scan_output = self.norm(scan_output, gate)
        out = self.out_proj(scan_output)

        return out


# ==========================================
# 3. RMSNorm (from fla.modules.layernorm)
# ==========================================


# ==========================================
# 4. PoST Mamba-2 CausalLM
# ==========================================

class Mamba2PoSTForCausalLM(PreTrainedModel, GenerationMixin):
    """Mamba-2 language model with PoST position-adaptive Δ-scaling."""
    config_class = Mamba2PoSTConfig
    _tied_weights_keys = {"lm_head.weight": "embedding.weight"}

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
        }

    def __init__(self, config: Mamba2PoSTConfig):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                mixer=Mamba2PoST(
                    num_heads=config.num_heads,
                    head_dim=config.head_dim,
                    hidden_size=config.hidden_size,
                    state_size=config.state_size,
                    expand=config.expand,
                    n_groups=config.n_groups,
                    conv_kernel=config.conv_kernel,
                    use_conv_bias=config.use_conv_bias,
                    hidden_act=config.hidden_act,
                    rms_norm=config.rms_norm,
                    chunk_size=config.chunk_size,
                    time_step_rank=config.time_step_rank,
                    time_step_limit=config.time_step_limit,
                    time_step_min=config.time_step_min,
                    time_step_max=config.time_step_max,
                    use_bias=config.use_bias,
                    norm_eps=config.norm_eps,
                    layer_idx=i,
                    train_length=config.train_length,
                    use_post=config.use_post,
                ),
                norm=RMSNorm(config.hidden_size),
            ))
            for i in range(config.num_hidden_layers)
        ])

        self.norm_f = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Explicit weight tying (HF's tie_weights via post_init doesn't work reliably)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self.post_init()

    def _init_weights(self, module):
        """Weight initialization matching FLA's Mamba2PreTrainedModel._init_weights.

        NOTE: A_log, D, and dt_bias are NOT re-initialized here.
        They are already properly initialized in Mamba2.__init__ (for vanilla)
        or Mamba2PoST.__init__ (for PoST). Re-initializing them here would
        break from_pretrained(), which calls _init_weights AFTER loading
        checkpoint weights — overwriting trained values with random ones.
        """
        n_layer = self.config.num_hidden_layers

        # --- Mamba2 layer: only set metadata attributes ---
        if isinstance(module, Mamba2PoST):
            if hasattr(module, 'A_log'):
                module.A_log._no_weight_decay = True
            if hasattr(module, 'A_log_base'):
                module.A_log_base._no_weight_decay = True
            if hasattr(module, 'A_log_deltas'):
                module.A_log_deltas._no_weight_decay = True
            module.D._no_weight_decay = True
            module.dt_bias._no_weight_decay = True

        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        # --- GPT-2 style residual rescaling (FLA: rescale_prenorm_residual=True) ---
        # Scale out_proj by 1/sqrt(n_layer) so residual accumulation is stable
        if hasattr(module, 'out_proj'):
            nn.init.kaiming_uniform_(module.out_proj.weight, a=math.sqrt(5))
            with torch.no_grad():
                module.out_proj.weight /= math.sqrt(n_layer)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids, labels=None, attention_mask=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        x = self.embedding(input_ids)
        for layer in self.layers:
            y = layer['mixer'](layer['norm'](x))
            x = y + x

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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
