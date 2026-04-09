"""
Debug the A_scale correctness — check where old vs new approach diverge.
Focus on verifying dA_cumsum is the same in both approaches.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd as cumsum_fwd_orig
from post_kernels.ssd_chunk_state import _chunk_cumsum_fwd as cumsum_fwd_new

torch.manual_seed(42)

batch, seqlen, nheads = 2, 128, 4
chunk_size = 64

# Simple inputs
dt_raw = torch.randn(batch, seqlen, nheads, device="cuda", dtype=torch.float32)
A = -torch.rand(nheads, device="cuda", dtype=torch.float32) * 0.1  # small A
dt_bias = torch.randn(nheads, device="cuda", dtype=torch.float32)

# Scale factors
positions = torch.arange(1, 1 + seqlen, device="cuda", dtype=torch.float32)
alpha = torch.tensor([0.1, 0.2, 0.3, 0.4], device="cuda", dtype=torch.float32)
scale = positions.unsqueeze(-1) ** (-alpha.unsqueeze(0))  # (seqlen, nheads)
A_scale_batch = scale.unsqueeze(0).expand(batch, -1, -1).contiguous()

# Apply softplus manually
dt = F.softplus(dt_raw + dt_bias)

# --- Old approach: scale dt ---
dt_scaled = dt * scale.unsqueeze(0)
dA_cumsum_old, dt_out_old = cumsum_fwd_orig(dt_scaled, A, chunk_size, dt_softplus=False)

# --- New approach: A_scale ---
dA_cumsum_new, dt_out_new = cumsum_fwd_new(dt.clone(), A.clone(), chunk_size, dt_softplus=False, A_scale=A_scale_batch)

# Check dA_cumsum (should be the same!)
# Old: dA = dt_scaled * A = dt * scale * A
# New: dA = dt * A * A_scale = dt * A * scale
# These should be identical since multiplication is commutative!
max_diff_dA = (dA_cumsum_old - dA_cumsum_new).abs().max().item()
print(f"dA_cumsum max diff: {max_diff_dA:.2e}")

# Check dt_out (should be DIFFERENT!)
# Old: dt_out stores dt_scaled (dt * scale)
# New: dt_out stores dt (unscaled)
max_diff_dt = (dt_out_old - dt_out_new).abs().max().item()
print(f"dt_out max diff: {max_diff_dt:.2e}")

# The dt_out goes into _chunk_state_fwd and _chunk_scan_fwd for input gain!  
# In the old approach, dt_out = dt * scale, so input gain would be (dt*scale) * B * x
# That's why the old approach compensates x by 1/scale → input = (dt*scale) * B * (x/scale) = dt * B * x
# In the new approach, dt_out = dt, so input = dt * B * x directly. Correct!
# BUT since the dt_out differs between old & new, the ENTIRE output differs, not just the decay.
# The test must take this into account.

print("\n--- Conclusion ---")
print(f"dA_cumsum (decay) match: {'YES' if max_diff_dA < 1e-5 else 'NO'} (diff={max_diff_dA:.2e})")
print(f"dt_out (input gain) differ: {'YES (expected!)' if max_diff_dt > 0.01 else 'NO (unexpected!)'} (diff={max_diff_dt:.2e})")
print("\nThe test should compare:")
print("  Old: mamba_scan(x_comp=x/s, dt_scaled=dt*s, D=None) + D*x")
print("  New: mamba_scan(x, dt, A_scale=s, D=D)")
print("Since dt_out differs, the x compensation IS needed for the old approach to match the new approach.")
