"""
PoST Models Package.

Re-exports all model classes for clean imports:
    from models import Mamba2PoSTConfig, Mamba2PoSTForCausalLM
    from models import RWKV7PoSTForCausalLM, RWKV7Config
    from models import GatedDeltaNetConfig, GDNPoSTForCausalLM
"""

# Mamba-2 PoST
from models.post_mamba2 import (
    Mamba2PoSTConfig,
    Mamba2PoST,
    Mamba2PoSTForCausalLM,
    RMSNorm,
)

# RWKV-7 PoST (requires fla)
try:
    from models.post_rwkv7 import (
        RWKV7Config,
        RWKV7PoSTAttention,
        RWKV7PoSTForCausalLM,
    )
except ImportError:
    RWKV7Config = None
    RWKV7PoSTAttention = None
    RWKV7PoSTForCausalLM = None

# Gated DeltaNet PoST (requires fla)
try:
    from models.post_gated_deltanet import (
        GatedDeltaNetConfig,
        GDNPoST,
        GDNPoSTForCausalLM,
    )
except ImportError:
    GatedDeltaNetConfig = None
    GDNPoST = None
    GDNPoSTForCausalLM = None

__all__ = [
    # Mamba-2 PoST
    "Mamba2PoSTConfig",
    "Mamba2PoST",
    "Mamba2PoSTForCausalLM",
    "RMSNorm",
    # RWKV-7
    "RWKV7Config",
    "RWKV7PoSTAttention",
    "RWKV7PoSTForCausalLM",
    # Gated DeltaNet
    "GatedDeltaNetConfig",
    "GDNPoST",
    "GDNPoSTForCausalLM",
]

