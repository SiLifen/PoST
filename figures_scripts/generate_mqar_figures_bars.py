#!/usr/bin/env python3
"""
Generate publication-quality MQAR **bar chart** figures for the PoST paper.

For each model dimension d ∈ {256, 128, 64}, produces one grouped bar plot
showing capacity accuracy (kv = T/4) at T ∈ {512, 1024, 2048, 8192}.

Data is fetched from wandb and cached locally (reuses the same cache as the
line-plot version).

Usage:
  python generate_mqar_figures_bars.py               # fetch from wandb + plot
  python generate_mqar_figures_bars.py --use-cached   # skip wandb, use cached json
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figs")
CACHE_FILE = os.path.join(SCRIPT_DIR, "mqar_wandb_data_all.json")

WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECTS = [
    "post-mqar-state-equalized",         # primary: state-equalized runs (renamed)
]

# ============================================================
# Experiment layout
# ============================================================
TRAIN_SEQ_LEN = 512

# Capacity test: kv = T/4 for each T
#   T=512  → kv=128
#   T=1024 → kv=256
#   T=2048 → kv=512
#   T=4096 → kv=1024
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
# Publication-quality style
# ============================================================

# Color families (same as line-plot version):
#   Mamba-2 style (green):  Mamba-2 = lighter green,   Mamba-2 PoST = darker green
#   RWKV style (blue):     RWKV-7  = lighter blue,    RWKV-7 PoST = darker blue
#   GDN style (orange):    GDN     = lighter orange,   GDN PoST   = darker orange
#   Attention:             red
MODEL_STYLE = {
    "Attention":        {"color": "#c0392b", "marker": "D", "ls": ":",  "ms": 7, "lw": 2.0, "zorder": 2},
    "RWKV-7":           {"color": "#7fb3d8", "marker": "o", "ls": "--", "ms": 7, "lw": 2.0, "zorder": 3},
    "RWKV-7 PoST":     {"color": "#1a5276", "marker": "s", "ls": "-",  "ms": 7, "lw": 2.2, "zorder": 4},
    "Mamba-2":          {"color": "#82e0aa", "marker": "^", "ls": "--", "ms": 7, "lw": 2.0, "zorder": 3},
    "Mamba-2 PoST":    {"color": "#1e8449", "marker": "v", "ls": "-",  "ms": 7, "lw": 2.2, "zorder": 5},
    "GDN":              {"color": "#f0b27a", "marker": "p", "ls": "--", "ms": 7, "lw": 2.0, "zorder": 3},
    "GDN PoST":        {"color": "#d35400", "marker": "h", "ls": "-",  "ms": 7, "lw": 2.2, "zorder": 5},
    "GLA":              {"color": "#bb8fce", "marker": "d", "ls": "--", "ms": 7, "lw": 2.0, "zorder": 3},
    "GLA PoST":        {"color": "#6c3483", "marker": "P", "ls": "-",  "ms": 7, "lw": 2.2, "zorder": 5},
    "RetNet":           {"color": "#f1948a", "marker": ">", "ls": "--", "ms": 7, "lw": 2.0, "zorder": 3},
    # RetNet PoST reuses gla_post data but keeps RetNet family color
    "RetNet PoST":     {"color": "#c0392b", "marker": "<", "ls": "-",  "ms": 7, "lw": 2.2, "zorder": 5},
}

# Plot order: baselines first, then PoST variants
PLOT_ORDER = [
    "Mamba-2 PoST", "Mamba-2", "RWKV-7 PoST", "RWKV-7", "GDN PoST", "GDN",
    "GLA PoST", "GLA", "RetNet PoST", "RetNet",
]

def setup_rcparams():
    """Set up matplotlib rcParams for publication-quality plots."""
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


# ============================================================
# wandb data fetching
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
        "valid/input_seq_len/accuracy-8192",
        "valid/num_kv_pairs/accuracy-128",
        "valid/num_kv_pairs/accuracy-256",
        "valid/num_kv_pairs/accuracy-512",
        "valid/num_kv_pairs/accuracy-2048",
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
# Data extraction
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

    # Extract accuracy per T (1× uses input_seq_len metric; others use capacity)
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
# Plotting — grouped bar chart version
# ============================================================

def plot_mqar_bars(ax, seq_lens, model_data, ylabel="Accuracy", ylim=(-0.02, 1.05)):
    """Plot grouped bars: one group per seq_len, one bar per model."""
    # Figure out which models actually have data
    active_models = []
    for model_name in PLOT_ORDER:
        data = model_data.get(model_name)
        if data is not None and any(data.get(sl) is not None for sl in seq_lens):
            active_models.append(model_name)

    n_models = len(active_models)
    if n_models == 0:
        return

    n_groups = len(seq_lens)
    bar_width = 0.8 / n_models  # total group width ≈ 0.8
    x = np.arange(n_groups)

    for i, model_name in enumerate(active_models):
        data = model_data[model_name]
        accs = [data.get(sl, 0) if data.get(sl) is not None else 0 for sl in seq_lens]
        s = MODEL_STYLE[model_name]
        offset = (i - (n_models - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            accs,
            width=bar_width * 0.88,  # slight gap between bars
            color=s["color"],
            label=model_name,
            edgecolor="white",
            linewidth=0.5,
            zorder=s["zorder"],
            alpha=0.92,
        )

    x_labels_full = [f"{sl}\n({sl // TRAIN_SEQ_LEN}×)" for sl in seq_lens]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels_full)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Context Length")
    ax.set_ylim(ylim)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


def generate_figures(all_data: dict):
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_rcparams()

    state_order = list(STATE_CONFIGS.keys())  # ["64K", "32K", "16K"]

    # ---- Collect data for all state configs ----
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

    # ---- Individual PDFs ----
    for state_label in state_order:
        best = all_best[state_label]
        cfg  = STATE_CONFIGS[state_label]
        has_any = any(v is not None for v in best.values())
        if not has_any:
            print(f"  ⏭ Skipping state={state_label} — no data\n")
            continue
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        cap_data = {m: r["capacity_per_len"] if r else None for m, r in best.items()}
        plot_mqar_bars(ax, SEQ_LENS, cap_data)
        ax.set_title(f"MQAR  ($d={cfg['dim']}$,  state$={state_label}$)")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG_DIR, f"mqar_bars_state{state_label}.{ext}"))
        print(f"  ✅ mqar_bars_state{state_label}\n")
        plt.close(fig)

    # ---- Combined 1×3 figure with shared legend ----
    states_with_data = [s for s in state_order
                        if any(v is not None for v in all_best[s].values())]
    if not states_with_data:
        print("  ⏭ No data for combined figure")
        return

    n = len(states_with_data)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 3.8), sharey=True)
    if n == 1:
        axes = [axes]

    for i, state_label in enumerate(states_with_data):
        ax = axes[i]
        best = all_best[state_label]
        cfg  = STATE_CONFIGS[state_label]
        cap_data = {m: r["capacity_per_len"] if r else None for m, r in best.items()}
        plot_mqar_bars(ax, SEQ_LENS, cap_data, ylabel="Accuracy" if i == 0 else "")
        ax.set_title(f"$d={cfg['dim']}$,  state$={state_label}$")
        if i > 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    ncol = min(len(labels), 5)
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, -0.06), frameon=True, borderaxespad=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"mqar_bars_combined.{ext}"))
    print(f"  ✅ mqar_bars_combined\n")
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

    print("\nGenerating bar chart figures...")
    generate_figures(all_data)
    print("Done.")


if __name__ == "__main__":
    main()
