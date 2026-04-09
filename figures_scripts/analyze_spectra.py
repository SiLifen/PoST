import argparse
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import rcParams
from matplotlib.lines import Line2D

def extract_taus(path, is_post, num_layers):
    state_dict = load_file(path)
    all_taus = []
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
                all_taus.append(torch.exp(-A_log).numpy())
            else:
                all_taus.append(np.ones(1)) # dummy
        else:
            A_log_key = f"layers.{l_idx}.mixer.A_log"
            if A_log_key not in state_dict:
                A_log_key = f"model.{A_log_key}"
            if A_log_key in state_dict:
                A_log = state_dict[A_log_key].float()
                all_taus.append(torch.exp(-A_log).numpy())
            else:
                 all_taus.append(np.ones(1))
    return np.array(all_taus)

def setup_rcparams():
    rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 12,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": True,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

def plot_density_and_gaps(t_v_180, t_p_180, t_v_440, t_p_440):
    sns.set_style("whitegrid")
    setup_rcparams()
    
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    
    # Facet style colors (Blue for 180M, Orange for 440M)
    c_180_post = "#1f77b4" # Strong blue
    c_180_vanilla = "#a6cce3" # Very light blue
    
    c_440_post = "#ff7f0e" # Strong orange
    c_440_vanilla = "#ffbb78" # Very light orange
    
    # --- Plot 1: Kernel Density Estimate of all Taus (Spectral Distribution) ---
    ax = axes[0]
    bins = np.logspace(np.log10(1e-3), np.log10(100), 100)
    
    sns.kdeplot(t_v_180.flatten(), color=c_180_vanilla, linestyle="--", label="Mamba-2 180M", ax=ax, log_scale=True, bw_adjust=0.5, linewidth=1.5)
    sns.kdeplot(t_p_180.flatten(), color=c_180_post, linestyle="-", label="Mamba-2 PoST 180M", ax=ax, log_scale=True, bw_adjust=0.5, linewidth=2.0)
    
    sns.kdeplot(t_v_440.flatten(), color=c_440_vanilla, linestyle="--", label="Mamba-2 440M", ax=ax, log_scale=True, bw_adjust=0.5, linewidth=1.5)
    sns.kdeplot(t_p_440.flatten(), color=c_440_post, linestyle="-", label="Mamba-2 PoST 440M", ax=ax, log_scale=True, bw_adjust=0.5, linewidth=2.0)
    
    ax.set_xlabel(r'Timescale $\tau_k = e^{-A_k}$')
    ax.set_ylabel('Density of SSM Heads')
    ax.set_title('Spectral Distribution (All Layers)')
    
    # --- Plot 2: Example Layer Progression (Layer 12) ---
    ax = axes[1]
    l_idx = 12
    # Sort slow → fast
    tv_180_l = np.sort(t_v_180[l_idx])[::-1]
    tp_180_l = np.sort(t_p_180[l_idx])[::-1]
    tv_440_l = np.sort(t_v_440[l_idx])[::-1]
    tp_440_l = np.sort(t_p_440[l_idx])[::-1]

    # Use actual 1-indexed head numbers (180M: 24 heads, 440M: 32 heads)
    x_180 = np.arange(1, len(tv_180_l) + 1)
    x_440 = np.arange(1, len(tv_440_l) + 1)

    ax.plot(x_180, tv_180_l, marker='o', markersize=4, color=c_180_vanilla, linestyle="--", linewidth=1.5, label="Mamba-2 180M")
    ax.plot(x_180, tp_180_l, marker='s', markersize=4, color=c_180_post,    linestyle="-",  linewidth=2.0, label="Mamba-2 PoST 180M")
    ax.plot(x_440, tv_440_l, marker='o', markersize=4, color=c_440_vanilla, linestyle="--", linewidth=1.5, label="Mamba-2 440M")
    ax.plot(x_440, tp_440_l, marker='s', markersize=4, color=c_440_post,    linestyle="-",  linewidth=2.0, label="Mamba-2 PoST 440M")

    ax.set_yscale('log')
    ax.set_xlabel('Head Index (slow \u2192 fast)')
    ax.set_ylabel(r'Timescale $\tau_k = e^{-A_k}$')
    ax.set_title('Head Allocations (Layer 12)')
    
    # Extract handles from the second axis because it has both markers and lines
    handles, labels = axes[1].get_legend_handles_labels()
    
    # Create a single unified legend at the bottom (2x2 grid)
    # Transposed: 
    # Row 1: Mamba-2 PoST 180M, Mamba-2 180M
    # Row 2: Mamba-2 PoST 440M, Mamba-2 440M
    unified_handles = [
        Line2D([0], [0], color=c_180_post, linestyle="-", linewidth=2.2, marker="s", markersize=7),
        Line2D([0], [0], color=c_180_vanilla, linestyle="--", linewidth=2.0, marker="o", markersize=7),
        Line2D([0], [0], color=c_440_post, linestyle="-", linewidth=2.2, marker="s", markersize=7),
        Line2D([0], [0], color=c_440_vanilla, linestyle="--", linewidth=2.0, marker="o", markersize=7)
    ]
    unified_labels = [
        "Mamba-2 PoST 180M",
        "Mamba-2 180M",
        "Mamba-2 PoST 440M",
        "Mamba-2 440M"
    ]
    
    fig.legend(unified_handles, unified_labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06), frameon=True,
               borderaxespad=0.3)
    
    # Leave room at bottom for the 2-row legend
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figs')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'post_spectra_analysis.pdf'), bbox_inches='tight')
    print("Saved dual analysis plot")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze spectral distribution of SSM timescales.")
    parser.add_argument("--model_dir_180m", type=str, required=True,
                        help="Path to 180M checkpoint directory (contains mamba2/ and mamba2_post/)")
    parser.add_argument("--model_dir_440m", type=str, required=True,
                        help="Path to 440M checkpoint directory (contains mamba2/ and mamba2_post/)")
    args = parser.parse_args()

    t_v_180 = extract_taus(os.path.join(args.model_dir_180m, "mamba2/mamba2/model.safetensors"), False, 24)
    t_p_180 = extract_taus(os.path.join(args.model_dir_180m, "mamba2_post/mamba2_post/model.safetensors"), True, 24)

    t_v_440 = extract_taus(os.path.join(args.model_dir_440m, "mamba2/mamba2/model.safetensors"), False, 48)
    t_p_440 = extract_taus(os.path.join(args.model_dir_440m, "mamba2_post/mamba2_post/model.safetensors"), True, 48)

    plot_density_and_gaps(t_v_180, t_p_180, t_v_440, t_p_440)

