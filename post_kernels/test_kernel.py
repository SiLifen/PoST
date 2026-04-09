"""
Test script for modified Mamba2 triton kernel with A_scale support.

Tests:
1. Backward compatibility: A_scale=None gives same result as original kernel
2. dA_cumsum correctness: A_scale in kernel produces same dA_cumsum as manually scaling dt
3. Gradient flow: A_scale gradients are non-zero and finite
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

# Original kernel (unchanged)
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as mamba_chunk_scan_combined_orig
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd as cumsum_fwd_orig
# Modified kernel (with A_scale support)
from post_kernels.ssd_combined import mamba_chunk_scan_combined as mamba_chunk_scan_combined_new
from post_kernels.ssd_chunk_state import _chunk_cumsum_fwd as cumsum_fwd_new


def make_inputs(batch=2, seqlen=256, nheads=8, headdim=16, ngroups=1, dstate=16, device="cuda", dtype=torch.float32):
    """Create random inputs for mamba_chunk_scan_combined."""
    x = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    dt = torch.randn(batch, seqlen, nheads, device=device, dtype=torch.float32)
    A = -torch.rand(nheads, device=device, dtype=torch.float32)
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype)
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype)
    D = torch.randn(nheads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(nheads, device=device, dtype=torch.float32)
    return x, dt, A, B, C, D, dt_bias


def test_backward_compatibility():
    """Test that A_scale=None gives identical results to original kernel."""
    print("=" * 60)
    print("Test 1: Backward compatibility (A_scale=None)")
    print("=" * 60)

    torch.manual_seed(42)
    x, dt, A, B, C, D, dt_bias = make_inputs()
    chunk_size = 64

    out_orig = mamba_chunk_scan_combined_orig(
        x.detach(), dt.clone(), A.clone(), B.clone(), C.clone(), chunk_size,
        D=D.clone(), dt_bias=dt_bias.clone(), dt_softplus=True,
    )

    out_new = mamba_chunk_scan_combined_new(
        x.detach(), dt.clone(), A.clone(), B.clone(), C.clone(), chunk_size,
        D=D.clone(), dt_bias=dt_bias.clone(), dt_softplus=True,
        A_scale=None,
    )

    max_diff = (out_orig - out_new).abs().max().item()
    print(f"  Max difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Backward compatibility failed! Max diff: {max_diff}"
    print("  ✓ PASSED\n")


def test_dA_cumsum_correctness():
    """Test that A_scale in kernel produces same dA_cumsum as scaling dt manually.

    dA_cumsum = cumsum(dt * A * A_scale)  via kernel
    should equal
    dA_cumsum = cumsum(dt_scaled * A)     where dt_scaled = dt * A_scale

    This verifies the core mathematical equivalence at the cumsum level.
    """
    print("=" * 60)
    print("Test 2: dA_cumsum correctness (A_scale vs dt-scaling)")
    print("=" * 60)

    torch.manual_seed(42)
    batch, seqlen, nheads = 2, 256, 8
    chunk_size = 64

    dt = torch.randn(batch, seqlen, nheads, device="cuda", dtype=torch.float32)
    A = -torch.rand(nheads, device="cuda", dtype=torch.float32)

    positions = torch.arange(1, 1 + seqlen, device="cuda", dtype=torch.float32)
    alpha = torch.rand(nheads, device="cuda") * 0.5
    scale = positions.unsqueeze(-1) ** (-alpha.unsqueeze(0))  # (seqlen, nheads)
    A_scale_batch = scale.unsqueeze(0).expand(batch, -1, -1).contiguous()

    dt_presoftplus = F.softplus(dt)

    # Old: scale dt, then cumsum
    dt_scaled = dt_presoftplus * scale.unsqueeze(0)
    dA_cumsum_old, _ = cumsum_fwd_orig(dt_scaled, A, chunk_size, dt_softplus=False)

    # New: A_scale in kernel
    dA_cumsum_new, _ = cumsum_fwd_new(dt_presoftplus.clone(), A.clone(), chunk_size, dt_softplus=False, A_scale=A_scale_batch)

    max_diff = (dA_cumsum_old - dA_cumsum_new).abs().max().item()
    print(f"  Max difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"dA_cumsum correctness failed! Max diff: {max_diff}"
    print("  ✓ PASSED\n")


def test_gradient_flow():
    """Test that gradients flow through A_scale parameter."""
    print("=" * 60)
    print("Test 3: Gradient flow through A_scale")
    print("=" * 60)

    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 128, 4, 16
    ngroups, dstate = 1, 16
    chunk_size = 64

    x = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.float32, requires_grad=True)
    dt = torch.randn(batch, seqlen, nheads, device="cuda", dtype=torch.float32)
    A = -torch.rand(nheads, device="cuda", dtype=torch.float32, requires_grad=True)
    B = torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=torch.float32)
    C = torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=torch.float32)
    D = torch.randn(nheads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(nheads, device="cuda", dtype=torch.float32)

    A_scale = torch.ones(batch, seqlen, nheads, device="cuda", dtype=torch.float32, requires_grad=True)

    out = mamba_chunk_scan_combined_new(
        x, dt, A, B, C, chunk_size,
        D=D, dt_bias=dt_bias, dt_softplus=True,
        A_scale=A_scale,
    )

    loss = out.sum()
    loss.backward()

    print(f"  A_scale.grad shape: {A_scale.grad.shape}")
    print(f"  A_scale.grad abs max: {A_scale.grad.abs().max().item():.2e}")
    print(f"  A_scale.grad abs mean: {A_scale.grad.abs().mean().item():.2e}")
    print(f"  x.grad abs max: {x.grad.abs().max().item():.2e}")

    assert A_scale.grad is not None, "A_scale.grad is None!"
    assert torch.isfinite(A_scale.grad).all(), "A_scale.grad has non-finite values!"
    assert A_scale.grad.abs().max().item() > 0, "A_scale.grad is all zeros!"
    print("  ✓ PASSED\n")


def test_full_pipeline():
    """Test full forward+backward with realistic A_scale values (like PoST)."""
    print("=" * 60)
    print("Test 4: Full pipeline with PoST-style A_scale")
    print("=" * 60)

    torch.manual_seed(42)
    batch, seqlen, nheads, headdim = 2, 256, 8, 16
    ngroups, dstate = 1, 16
    chunk_size = 64

    x = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.float32, requires_grad=True)
    dt = torch.randn(batch, seqlen, nheads, device="cuda", dtype=torch.float32)
    A = -torch.rand(nheads, device="cuda", dtype=torch.float32, requires_grad=True)
    B = torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=torch.float32)
    C = torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=torch.float32)
    D = torch.randn(nheads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(nheads, device="cuda", dtype=torch.float32)

    # PoST-style A_scale = t^(-alpha)
    positions = torch.arange(1, 1 + seqlen, device="cuda", dtype=torch.float32)
    alpha = torch.rand(nheads, device="cuda") * 0.5
    A_scale = (positions.unsqueeze(-1) ** (-alpha.unsqueeze(0))).unsqueeze(0).expand(batch, -1, -1).contiguous()
    A_scale.requires_grad_(True)

    out = mamba_chunk_scan_combined_new(
        x, dt, A, B, C, chunk_size,
        D=D, dt_bias=dt_bias, dt_softplus=True,
        A_scale=A_scale,
    )

    assert torch.isfinite(out).all(), "Output has non-finite values!"
    print(f"  Output shape: {out.shape}")
    print(f"  Output abs max: {out.abs().max().item():.2e}")

    loss = out.sum()
    loss.backward()

    assert torch.isfinite(x.grad).all(), "x.grad has non-finite values!"
    assert torch.isfinite(A_scale.grad).all(), "A_scale.grad has non-finite values!"
    print(f"  All gradients finite: ✓")
    print(f"  A_scale.grad range: [{A_scale.grad.min().item():.2e}, {A_scale.grad.max().item():.2e}]")
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running A_scale kernel tests")
    print("=" * 60 + "\n")

    test_backward_compatibility()
    test_dA_cumsum_correctness()
    test_gradient_flow()
    test_full_pipeline()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
