"""
plot_heatmap.py

Generates a 2×2 grid of Layer x Head heatmaps of log(τ_k) = -A_k
for (Baseline 180M, PoST 180M, Baseline 440M, PoST 440M).

Each heatmap shows how the decay timescales are distributed
across all heads and all layers of the model.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
import os


def extract_log_taus(path, is_post, num_layers):
    """
    Returns array of shape (num_layers, num_heads) with log(τ_k) = -log(A_k).
    """
    state_dict = load_file(path)
    rows = []
    for l_idx in range(num_layers):
        if is_post:
            base_key = f"layers.{l_idx}.mixer.A_log_base"
            deltas_key = f"layers.{l_idx}.mixer.A_log_deltas"
            if base_key not in state_dict:
                base_key = f"model.{base_key}"
                deltas_key = f"model.{deltas_key}"
            if base_key in state_dict:
                base = state_dict[base_key].float()
                deltas = state_dict[deltas_key].float()
                d = F.softplus(deltas)
                offsets = torch.cat([torch.zeros(1), torch.cumsum(d, dim=0)])
                A_log = base + offsets
                # log(τ_k) = -A_k  (A_k is already the log-decay, τ = e^{-A})
                log_tau = -A_log.numpy()
            else:
                log_tau = np.zeros(1)
        else:
            A_log_key = f"layers.{l_idx}.mixer.A_log"
            if A_log_key not in state_dict:
                A_log_key = f"model.{A_log_key}"
            if A_log_key in state_dict:
                A_log = state_dict[A_log_key].float()
                log_tau = -A_log.numpy()
            else:
                log_tau = np.zeros(1)
        # Keep raw head ordering: PoST is sorted by construction (cumulative softplus),
        # baseline is unsorted (no constraint). This makes the structural difference visible.
        rows.append(log_tau.copy())
    return np.array(rows)   # (num_layers, num_heads)


def setup_rcparams():
    rcParams.update({
        "mathtext.fontset": "stix",
        "font.family":      "STIXGeneral",
        "font.size":         11,
        "axes.labelsize":    11,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
    })


def plot_heatmaps(lt_v_180, lt_p_180, lt_v_440, lt_p_440):
    setup_rcparams()

    panels = [
        (lt_v_180, "Mamba-2 180M",        "Blues_r"),
        (lt_p_180, "Mamba-2 PoST 180M",   "Blues_r"),
        (lt_v_440, "Mamba-2 440M",         "Oranges_r"),
        (lt_p_440, "Mamba-2 PoST 440M",   "Oranges_r"),
    ]

    # Determine shared color range per model scale
    vmin_180 = min(lt_v_180.min(), lt_p_180.min())
    vmax_180 = max(lt_v_180.max(), lt_p_180.max())
    vmin_440 = min(lt_v_440.min(), lt_p_440.min())
    vmax_440 = max(lt_v_440.max(), lt_p_440.max())

    vmins = [vmin_180, vmin_180, vmin_440, vmin_440]
    vmaxs = [vmax_180, vmax_180, vmax_440, vmax_440]

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), constrained_layout=False)
    axes = axes.flatten()  # (top-left, top-right, bottom-left, bottom-right)

    ims = []
    for idx, (data, title, cmap) in enumerate(panels):
        ax = axes[idx]
        num_layers, num_heads = data.shape
        im = ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=vmins[idx],
            vmax=vmaxs[idx],
            interpolation="nearest",
        )
        ims.append(im)
        ax.set_title(title, pad=5)
        ax.set_xlabel("Head Index", labelpad=4)
        ax.set_ylabel("Layer", labelpad=4)
        ax.set_xticks(np.linspace(0, num_heads - 1, 5, dtype=int))
        ax.set_yticks(np.linspace(0, num_layers - 1, 6, dtype=int))

    # Shared colorbars: one per row (samemodel scale)
    plt.tight_layout(rect=[0, 0, 0.88, 1], h_pad=2.5, w_pad=2.0)

    # Colorbar for 180M row (top)
    cax180 = fig.add_axes([0.90, 0.56, 0.025, 0.36])
    cb180 = fig.colorbar(ims[0], cax=cax180)
    cb180.set_label(r"$\log\,\tau_k$", fontsize=11)

    # Colorbar for 440M row (bottom)
    cax440 = fig.add_axes([0.90, 0.10, 0.025, 0.36])
    cb440 = fig.colorbar(ims[2], cax=cax440)
    cb440.set_label(r"$\log\,\tau_k$", fontsize=11)

    fig_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figs"
    )
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "post_heatmap.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Layer x Head heatmaps of log(τ_k).")
    parser.add_argument("--model_dir_180m", type=str, required=True,
                        help="Path to 180M checkpoint directory (contains mamba2/ and mamba2_post/)")
    parser.add_argument("--model_dir_440m", type=str, required=True,
                        help="Path to 440M checkpoint directory (contains mamba2/ and mamba2_post/)")
    args = parser.parse_args()

    lt_v_180 = extract_log_taus(
        os.path.join(args.model_dir_180m, "mamba2/mamba2/model.safetensors"),
        False, 24)
    lt_p_180 = extract_log_taus(
        os.path.join(args.model_dir_180m, "mamba2_post/mamba2_post/model.safetensors"),
        True, 24)

    lt_v_440 = extract_log_taus(
        os.path.join(args.model_dir_440m, "mamba2/mamba2/model.safetensors"),
        False, 48)
    lt_p_440 = extract_log_taus(
        os.path.join(args.model_dir_440m, "mamba2_post/mamba2_post/model.safetensors"),
        True, 48)

    plot_heatmaps(lt_v_180, lt_p_180, lt_v_440, lt_p_440)
