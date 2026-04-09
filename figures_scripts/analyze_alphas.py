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

def get_A_log_post(base, deltas):
    d = F.softplus(deltas)
    zeros = torch.zeros(1, dtype=d.dtype)
    offsets = torch.cat([zeros, torch.cumsum(d, dim=0)])
    return base + offsets

def get_alpha_post(A_log, train_length=2048):
    cum = A_log - A_log[0]
    total = cum[-1]
    N = len(A_log)
    k = torch.arange(N, dtype=A_log.dtype)
    mean_gap = total / (N - 1)
    linear_taper = (N - 1 - k) / (N - 1)
    correction = (cum - k * mean_gap) / np.log(train_length)
    alpha = linear_taper + correction
    return alpha.clamp(min=0.0, max=1.0)

def extract_alphas(path, num_layers, is_post=True):
    state_dict = load_file(path)
    all_alphas = []
    
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
                A_log = get_A_log_post(base, deltas)
                alpha = get_alpha_post(A_log, 2048)
                all_alphas.append(alpha.numpy())
                
    # Vanilla Mamba-2 has no alpha tapers, so its effective alpha is 0
    if not is_post:
        # Get head count directly from A_log of the first layer if possible
        num_heads = 24 if num_layers == 24 else 32
        
        # We check the size from the state dict of the first layer if it exists
        try:
            A_log_vanilla = state_dict['layers.0.mixer.A_log']
            num_heads = A_log_vanilla.shape[0]
        except KeyError:
            try:
                A_log_vanilla = state_dict['model.layers.0.mixer.A_log']
                num_heads = A_log_vanilla.shape[0]
            except KeyError:
                pass
                
        all_alphas = np.zeros((num_layers, num_heads))
        
    return np.array(all_alphas)

def plot_alphas(a_p_180, a_v_180, a_p_440, a_v_440):
    sns.set_style("whitegrid")
    setup_rcparams()
    
    # Use exact MQAR dimensions
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    
    # MQAR Colors
    post_color_180 = "#1e8449"  # Dark green
    vanilla_color_180 = "#82e0aa" # Light green
    
    post_color_440 = "#d35400"  # Dark orange
    vanilla_color_440 = "#f0b27a" # Light orange
    
    # Plot 1: 180M vs Theoretical
    ax = axes[0]
    N = a_p_180.shape[1]
    x = np.linspace(0, 1, N)
    theoretical = np.linspace(1, 0, N)
    
    p_mean = np.mean(a_p_180, axis=0)
    p_std = np.std(a_p_180, axis=0)
    
    ax.plot(x, theoretical, color="black", linestyle=":", linewidth=2, label="Geometric Blueprint")
    
    ax.plot(x, p_mean, color=post_color_180, linestyle="-", linewidth=2.0, marker="v", markersize=4, label="Mamba-2 PoST 180M")
    ax.fill_between(x, p_mean - p_std, p_mean + p_std, color=post_color_180, alpha=0.2)
    
    ax.set_title('Taper Profile ($d=768$, 180M)')
    ax.set_xlabel('Normalized Head Ranking')
    ax.set_ylabel(r'Taper Exponent $\alpha_k$')
    
    # Plot 2: 440M vs Theoretical
    ax = axes[1]
    # Extract the actual number of heads for 440M (N=32)
    N_440 = a_p_440.shape[1]
    x_440 = np.linspace(0, 1, N_440)
    theoretical_440 = np.linspace(1, 0, N_440)
    
    p_mean = np.mean(a_p_440, axis=0)
    p_std = np.std(a_p_440, axis=0)
    
    ax.plot(x_440, theoretical_440, color="black", linestyle=":", linewidth=2, label="Geometric Blueprint")
    
    ax.plot(x_440, p_mean, color=post_color_440, linestyle="-", linewidth=2.0, marker="v", markersize=4, label="Mamba-2 PoST 440M")
    ax.fill_between(x_440, p_mean - p_std, p_mean + p_std, color=post_color_440, alpha=0.2)
    
    ax.set_title('Taper Profile ($d=1024$, 440M)')
    ax.set_xlabel('Normalized Head Ranking')
    
    # We want to display all 3 logical lines: Blueprint, PoST 180, PoST 440
    # Let's cleanly construct the unified legend manually
    unified_handles = [
        Line2D([0], [0], color=post_color_180, linestyle="-", linewidth=2.2, marker="v", markersize=7),
        Line2D([0], [0], color=post_color_440, linestyle="-", linewidth=2.2, marker="v", markersize=7),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2.0)
    ]
    unified_labels = [
        "Mamba-2 PoST 180M",
        "Mamba-2 PoST 440M",
        "Geometric Blueprint"
    ]
    
    fig.legend(unified_handles, unified_labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), frameon=True,
               borderaxespad=0.3)
    
    # Leave room at bottom for legend
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figs')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'post_alpha_comparison.pdf'), bbox_inches='tight')
    print("Saved alpha plot")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze learned alpha taper profiles.")
    parser.add_argument("--model_dir_180m", type=str, required=True,
                        help="Path to 180M checkpoint directory (contains mamba2/ and mamba2_post/)")
    parser.add_argument("--model_dir_440m", type=str, required=True,
                        help="Path to 440M checkpoint directory (contains mamba2/ and mamba2_post/)")
    args = parser.parse_args()

    a_p_180 = extract_alphas(os.path.join(args.model_dir_180m, "mamba2_post/mamba2_post/model.safetensors"), 24, is_post=True)
    a_v_180 = extract_alphas(os.path.join(args.model_dir_180m, "mamba2/mamba2/model.safetensors"), 24, is_post=False)

    a_p_440 = extract_alphas(os.path.join(args.model_dir_440m, "mamba2_post/mamba2_post/model.safetensors"), 48, is_post=True)
    a_v_440 = extract_alphas(os.path.join(args.model_dir_440m, "mamba2/mamba2/model.safetensors"), 48, is_post=False)

    plot_alphas(a_p_180, a_v_180, a_p_440, a_v_440)

