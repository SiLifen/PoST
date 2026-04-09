#!/usr/bin/env python3
"""
Generate publication-quality MQAR **faceted** figures for the PoST paper.

Layout: 3×3 grid
  - Rows    = model dimension d ∈ {512, 256, 128}
  - Columns = model family {Mamba-2, RWKV-7, GDN}
  - Each cell shows 2 lines: base model (dashed) vs PoST (solid)

This avoids the overlap problem of cramming 6 lines into one panel.

Usage:
  python generate_mqar_figures_faceted.py               # fetch from wandb + plot
  python generate_mqar_figures_faceted.py --use-cached   # skip wandb, use cached json
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figs")
CACHE_FILE = os.path.join(SCRIPT_DIR, "mqar_wandb_data_all.json")

WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECTS = [
    "post-mqar-state-equalized",
]

# ============================================================
# Experiment layout
# ============================================================
TRAIN_SEQ_LEN = 512

SEQ_LENS = [512, 1024, 2048, 4096]
KV_FOR_SEQ = {512: 128, 1024: 256, 2048: 512, 4096: 1024}
# For the 1× point (T=512), use input_seq_len metric to avoid mixing in
# OOD persist tests at longer lengths (which also have num_kv_pairs=128).
METRIC_FOR_SEQ = {
    512:  "valid/input_seq_len/accuracy-512",   # in-distribution only
    1024: "valid/num_kv_pairs/accuracy-256",     # capacity @ 2×
    2048: "valid/num_kv_pairs/accuracy-512",     # capacity @ 4×
    4096: "valid/num_kv_pairs/accuracy-1024",    # capacity @ 8×
}

_MODELS = [
    ("RWKV-7",       "rwkv7",      [10**-4.5, 1e-4, 10**-3.5], "rwkv"),
    ("RWKV-7 PoST",  "rwkv7_post", [10**-4.5, 1e-4, 10**-3.5], "rwkv"),
    ("Mamba-2",      "mamba2",     [1e-3, 10**-2.5, 1e-2],     None),
    ("Mamba-2 PoST", "mamba2_post",[1e-3, 10**-2.5, 1e-2],     None),
    ("GDN",          "gdn",        [10**-3.5, 1e-3, 10**-2.5], None),
    ("GDN PoST",     "gdn_post",   [10**-3.5, 1e-3, 10**-2.5], None),
    ("GLA",          "gla",        [1e-3, 10**-2.5, 1e-2],     None),
    ("GLA PoST",     "gla_post",   [1e-3, 10**-2.5, 1e-2],     None),
    ("RetNet",       "retnet",     [1e-3, 10**-2.5, 1e-2],     None),
    # RetNet PoST reuses the gla_post runs (same experiment)
    ("RetNet PoST",  "gla_post",   [1e-3, 10**-2.5, 1e-2],     None),
]

STATE_CONFIGS = {
    "64K": {"dim": 512, "heads": 4, "rwkv_heads": 4, "models": _MODELS},
    "32K": {"dim": 512, "heads": 8, "rwkv_heads": 8, "models": _MODELS},
    "16K": {"dim": 256, "heads": 4, "rwkv_heads": 4, "models": _MODELS},
}

# ============================================================
# Style — same colours as original
# ============================================================
MODEL_STYLE = {
    "Attention":        {"color": "#c0392b", "marker": "D", "ls": ":",  "ms": 5, "lw": 1.3, "zorder": 2},
    "RWKV-7":           {"color": "#7fb3d8", "marker": "o", "ls": "--", "ms": 5, "lw": 1.3, "zorder": 3},
    "RWKV-7 PoST":     {"color": "#1a5276", "marker": "s", "ls": "-",  "ms": 5, "lw": 1.5, "zorder": 4},
    "Mamba-2":          {"color": "#82e0aa", "marker": "^", "ls": "--", "ms": 5, "lw": 1.3, "zorder": 3},
    "Mamba-2 PoST":    {"color": "#1e8449", "marker": "v", "ls": "-",  "ms": 5, "lw": 1.5, "zorder": 5},
    "GDN":              {"color": "#f0b27a", "marker": "p", "ls": "--", "ms": 5, "lw": 1.3, "zorder": 3},
    "GDN PoST":        {"color": "#d35400", "marker": "h", "ls": "-",  "ms": 5, "lw": 1.5, "zorder": 5},
    "GLA":              {"color": "#bb8fce", "marker": "d", "ls": "--", "ms": 5, "lw": 1.3, "zorder": 3},
    "GLA PoST":        {"color": "#6c3483", "marker": "P", "ls": "-",  "ms": 5, "lw": 1.5, "zorder": 5},
    "RetNet":           {"color": "#f1948a", "marker": ">", "ls": "--", "ms": 5, "lw": 1.3, "zorder": 3},
    # RetNet PoST reuses gla_post data but keeps RetNet family color
    "RetNet PoST":     {"color": "#c0392b", "marker": "<", "ls": "-",  "ms": 5, "lw": 1.5, "zorder": 5},
}

# Model families: (column title, base model, PoST model)
MODEL_FAMILIES = [
    ("Mamba-2",  "Mamba-2",  "Mamba-2 PoST"),
    ("RWKV-7",   "RWKV-7",   "RWKV-7 PoST"),
    ("GDN",      "GDN",       "GDN PoST"),
    ("GLA",      "GLA",       "GLA PoST"),
    ("RetNet",   "RetNet",    "RetNet PoST"),
]

def setup_rcparams():
    """Compact but readable styling."""
    rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 8.5,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": True,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.3,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


# ============================================================
# wandb data fetching  (identical to original)
# ============================================================

def make_run_id(prefix: str, heads: int, dim: int, lr: float) -> str:
    return f"{prefix}-h{heads}-d{dim}-lr{lr:.1e}"


def fetch_wandb_data():
    import wandb
    api = wandb.Api()

    keys = [
        "valid/test_type/accuracy-capacity",
        "valid/test_type/accuracy-persist",
        "valid/accuracy",
        "valid/input_seq_len/accuracy-512",
        "valid/input_seq_len/accuracy-1024",
        "valid/input_seq_len/accuracy-2048",
        "valid/input_seq_len/accuracy-4096",
        "valid/num_kv_pairs/accuracy-128",
        "valid/num_kv_pairs/accuracy-256",
        "valid/num_kv_pairs/accuracy-512",
        "valid/num_kv_pairs/accuracy-1024",
        "epoch", "_step",
    ]

    all_data = {}
    for project in WANDB_PROJECTS:
        print(f"  Scanning project: {project}")
        all_runs = api.runs(f"{WANDB_ENTITY}/{project}", per_page=100)
        for run in all_runs:
            if run.state in ("finished", "running") and run.name not in all_data:
                print(f"  Fetching {run.name} ...")
                history = run.scan_history(keys=keys, page_size=1000)
                all_data[run.name] = [dict(row) for row in history]
                print(f"    {len(all_data[run.name])} epochs")
    return all_data


# ============================================================
# Data extraction  (identical to original)
# ============================================================

def select_best_run(all_data: dict, run_ids: list) -> dict | None:
    """Select best (run, epoch) by maximising sum of 4 capacity accuracies."""
    best_score = -1
    best_row = None
    best_rid = None

    for rid in run_ids:
        if rid not in all_data:
            continue
        for row in all_data[rid]:
            score = 0
            for sl in SEQ_LENS:
                v = row.get(METRIC_FOR_SEQ[sl])
                if v is not None:
                    score += v
            if score > best_score:
                best_score = score
                best_row = row
                best_rid = rid

    if best_row is None:
        return None

    capacity = {}
    for sl in SEQ_LENS:
        v = best_row.get(METRIC_FOR_SEQ[sl])
        if v is not None:
            capacity[sl] = v

    return {
        "run_id": best_rid,
        "epoch": best_row.get("epoch"),
        "score": best_score,
        "capacity_per_len": capacity,
    }


# ============================================================
# Plotting — faceted 3×3
# ============================================================

def plot_pair(ax, seq_lens, base_data, post_data, base_name, post_name,
              ylabel="", show_xticklabels=True):
    """Plot two lines (base vs PoST) on one compact axis."""
    x = np.arange(len(seq_lens))

    for model_name, data in [(base_name, base_data), (post_name, post_data)]:
        if data is None:
            continue
        accs = [data.get(sl, np.nan) for sl in seq_lens]
        s = MODEL_STYLE[model_name]
        ax.plot(
            x, accs,
            color=s["color"],
            marker=s["marker"],
            markersize=4.5,
            linewidth=s["lw"],
            linestyle=s["ls"],
            label=model_name,
            alpha=0.92,
            zorder=s["zorder"],
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

    # Shade the gap between base and PoST to highlight improvement
    if base_data is not None and post_data is not None:
        base_accs = np.array([base_data.get(sl, np.nan) for sl in seq_lens])
        post_accs = np.array([post_data.get(sl, np.nan) for sl in seq_lens])
        mask = ~(np.isnan(base_accs) | np.isnan(post_accs))
        if mask.any():
            # Use the PoST color for the fill
            post_color = MODEL_STYLE[post_name]["color"]
            ax.fill_between(
                x[mask], base_accs[mask], post_accs[mask],
                alpha=0.10, color=post_color, zorder=1,
            )

    if show_xticklabels:
        x_labels = [f"{sl}\n({sl // TRAIN_SEQ_LEN}×)" for sl in seq_lens]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([])

    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x", alpha=0.08)


def generate_figures(all_data: dict):
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_rcparams()

    state_order = list(STATE_CONFIGS.keys())  # ["64K", "32K", "16K"]

    # ---- Collect best runs ----
    all_best = {}
    for state_label, cfg in STATE_CONFIGS.items():
        dim        = cfg["dim"]
        heads      = cfg["heads"]
        rwkv_heads = cfg.get("rwkv_heads", heads)
        best = {}
        for model_name, prefix, lrs, model_type in cfg["models"]:
            h = rwkv_heads if model_type == "rwkv" else heads
            run_ids = [make_run_id(prefix, h, dim, lr) for lr in lrs]
            result = select_best_run(all_data, run_ids)
            best[model_name] = result
            if result:
                print(f"  {model_name} state={state_label}: {result['run_id']}, "
                      f"ep={result['epoch']}, score={result['score']:.3f}")
            else:
                print(f"  {model_name} state={state_label}: (pending)")
        all_best[state_label] = best

    # ---- Faceted figure (rows=families, cols=state sizes) ----
    n_rows = len(MODEL_FAMILIES)
    n_cols = len(state_order)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.2, 1.7 * n_rows),
        sharey=True, sharex=True,
    )

    for row_i, (family_name, base_name, post_name) in enumerate(MODEL_FAMILIES):
        for col_j, state_label in enumerate(state_order):
            ax = axes[row_i, col_j]
            best = all_best[state_label]
            cfg  = STATE_CONFIGS[state_label]

            base_result = best.get(base_name)
            post_result = best.get(post_name)
            base_cap = base_result["capacity_per_len"] if base_result else None
            post_cap = post_result["capacity_per_len"] if post_result else None

            is_bottom = (row_i == n_rows - 1)
            ylabel = family_name if col_j == 0 else ""

            plot_pair(
                ax, SEQ_LENS, base_cap, post_cap, base_name, post_name,
                ylabel=ylabel,
                show_xticklabels=is_bottom,
            )

            if row_i == 0:
                ax.set_title(f"$d={cfg['dim']}$,  state$={state_label}$")

            if base_cap is not None or post_cap is not None:
                ax.legend(loc="lower left", fontsize=7.5, handlelength=1.5,
                          borderpad=0.3, labelspacing=0.25, handletextpad=0.4)

    for col_j in range(n_cols):
        axes[-1, col_j].set_xlabel("Context Length")

    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"mqar_faceted.{ext}"))
    print(f"\n  ✅ mqar_faceted\n")
    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cached", action="store_true")
    args = parser.parse_args()

    if args.use_cached and os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}")
        with open(CACHE_FILE) as f:
            all_data = json.load(f)
    else:
        print("Fetching data from wandb...")
        all_data = fetch_wandb_data()
        with open(CACHE_FILE, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"Cached to {CACHE_FILE}")

    print("\nGenerating faceted figures...")
    generate_figures(all_data)
    print("Done.")


if __name__ == "__main__":
    main()
