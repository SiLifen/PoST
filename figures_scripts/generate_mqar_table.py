#!/usr/bin/env python3
"""
Regenerate the MQAR table for the paper with two fixes:
  1. 1× column uses valid/input_seq_len/accuracy-512 (unambiguously in-distribution)
     instead of valid/num_kv_pairs/accuracy-128 (which averages OOD persist tests too).
  2. Columns are 64K / 32K / 16K state sizes (matching current STATE_CONFIGS)
     instead of d=512 / d=256 / d=128.

Usage:
    python generate_mqar_table.py               # fetch from wandb + output
    python generate_mqar_table.py --use-cached  # use cached json
"""

import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE  = os.path.join(SCRIPT_DIR, "mqar_wandb_data_all.json")

WANDB_ENTITY   = os.environ.get("WANDB_ENTITY")
WANDB_PROJECTS = ["post-mqar-state-equalized"]

TRAIN_SEQ_LEN = 512
SEQ_LENS      = [512, 1024, 2048, 4096]

# For T=512 use input_seq_len metric (no OOD mixing); for others use capacity metric.
METRIC_FOR_SEQ = {
    512:  "valid/input_seq_len/accuracy-512",
    1024: "valid/num_kv_pairs/accuracy-256",
    2048: "valid/num_kv_pairs/accuracy-512",
    4096: "valid/num_kv_pairs/accuracy-1024",
}

_MODELS = [
    ("Mamba-2",      "Mamba-2 PoST",  "mamba2",     "mamba2_post",  [1e-3, 10**-2.5, 1e-2], None),
    ("RWKV-7",       "RWKV-7 PoST",   "rwkv7",      "rwkv7_post",   [10**-4.5, 1e-4, 10**-3.5], "rwkv"),
    ("GDN",          "GDN PoST",      "gdn",        "gdn_post",     [10**-3.5, 1e-3, 10**-2.5], None),
    ("GLA",          "GLA PoST",      "gla",        "gla_post",     [1e-3, 10**-2.5, 1e-2], None),
    ("RetNet",       "RetNet PoST",   "retnet",     "gla_post",     [1e-3, 10**-2.5, 1e-2], None),
]

# 64K → d=512, h=4   |   32K → d=512, h=8   |   16K → d=256, h=4
STATE_CONFIGS = {
    "64K": {"dim": 512, "heads": 4,  "rwkv_heads": 4},
    "32K": {"dim": 512, "heads": 8,  "rwkv_heads": 8},
    "16K": {"dim": 256, "heads": 4,  "rwkv_heads": 4},
}


# ── wandb ────────────────────────────────────────────────────────────────────

def fetch_wandb_data():
    import wandb
    api  = wandb.Api()
    keys = list(METRIC_FOR_SEQ.values()) + ["epoch", "_step"]
    all_data = {}
    for project in WANDB_PROJECTS:
        print(f"  Scanning {project} ...")
        for run in api.runs(f"{WANDB_ENTITY}/{project}", per_page=100):
            if run.state in ("finished", "running") and run.name not in all_data:
                print(f"    {run.name}")
                all_data[run.name] = [dict(r) for r in run.scan_history(keys=keys, page_size=1000)]
    return all_data


def make_run_id(prefix, heads, dim, lr):
    return f"{prefix}-h{heads}-d{dim}-lr{lr:.1e}"


# ── best-run selection ────────────────────────────────────────────────────────

def select_best_run(all_data, run_ids):
    best_score, best_row, best_rid = -1, None, None
    for rid in run_ids:
        if rid not in all_data:
            continue
        for row in all_data[rid]:
            score = sum(
                row.get(METRIC_FOR_SEQ[sl], 0) or 0
                for sl in SEQ_LENS
            )
            if score > best_score:
                best_score, best_row, best_rid = score, row, rid
    if best_row is None:
        return None
    accs = {sl: best_row.get(METRIC_FOR_SEQ[sl]) for sl in SEQ_LENS}
    return {"run_id": best_rid, "epoch": best_row.get("epoch"),
            "score": best_score, "accs": accs}


# ── formatting ────────────────────────────────────────────────────────────────

def fmt(v, bold=False):
    if v is None:
        return "---"
    s = f"{v*100:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def avg(accs):
    vals = [v for v in accs.values() if v is not None]
    return sum(vals) / len(vals) if vals else None


def row_cells(result_base, result_post, state_label):
    """Return (base_cells, post_cells) where each is a list of 5 strings.
    The winner is bolded; if both values are equal, both are bolded."""
    accs_b = result_base["accs"]  if result_base  else {sl: None for sl in SEQ_LENS}
    accs_p = result_post["accs"] if result_post else {sl: None for sl in SEQ_LENS}

    cells_b, cells_p = [], []
    for sl in SEQ_LENS:
        vb = accs_b.get(sl)
        vp = accs_p.get(sl)
        both_present = vb is not None and vp is not None
        pb = both_present and vp >= vb   # bold post if it wins or ties
        pp = both_present and vb >= vp   # bold base if it wins or ties
        cells_b.append(fmt(vb, bold=pp))
        cells_p.append(fmt(vp, bold=pb))

    ab, ap = avg(accs_b), avg(accs_p)
    both_avg = ab is not None and ap is not None
    cells_b.append(fmt(ab, bold=(both_avg and ab >= ap)))
    cells_p.append(fmt(ap, bold=(both_avg and ap >= ab)))
    return cells_b, cells_p


# ── main ─────────────────────────────────────────────────────────────────────

def generate_table(all_data):
    state_order = ["64K", "32K", "16K"]

    # collect results
    all_results = {}   # state_label -> {model_name -> result}
    for state_label in state_order:
        cfg = STATE_CONFIGS[state_label]
        dim, heads, rwkv_heads = cfg["dim"], cfg["heads"], cfg["rwkv_heads"]
        results = {}
        for base_name, post_name, base_prefix, post_prefix, lrs, model_type in _MODELS:
            h = rwkv_heads if model_type == "rwkv" else heads
            results[base_name] = select_best_run(all_data,
                [make_run_id(base_prefix, h, dim, lr) for lr in lrs])
            results[post_name] = select_best_run(all_data,
                [make_run_id(post_prefix, h, dim, lr) for lr in lrs])
            for name, res in [(base_name, results[base_name]),
                              (post_name, results[post_name])]:
                if res:
                    print(f"  {name:20s} state={state_label}: {res['run_id']}, "
                          f"ep={res['epoch']}, score={res['score']:.4f}")
                else:
                    print(f"  {name:20s} state={state_label}: (pending)")
        all_results[state_label] = results

    # ── LaTeX output ─────────────────────────────────────────────────────────
    col_header = " & ".join(
        f"\\multicolumn{{5}}{{c}}{{state$={s}$}}" for s in state_order
    )
    cmidrule = " ".join(
        f"\\cmidrule(lr){{{2+i*5}-{6+i*5}}}" for i in range(len(state_order))
    )
    sub_header = " & ".join(["512 & 1K & 2K & 4K & Avg"] * len(state_order))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{MQAR capacity accuracy (\%)} at each test length $T$ with $K{=}T/4$ key--value pairs.")
    lines.append(r"All models trained at $T{=}512$; longer lengths are out-of-distribution.")
    lines.append(r"The 1$\times$ column ($T{=}512$) reports in-distribution accuracy from the \texttt{input\_seq\_len} metric.")
    lines.append(r"GLA\,PoST and RetNet\,PoST converge to the same model under the PoST parameterization.}")
    lines.append(r"\label{tab:mqar}")
    lines.append(r"\small")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}l " + "ccccc " * len(state_order) + r"@{}}")
    lines.append(r"\toprule")
    lines.append(f" & {col_header} \\\\")
    lines.append(f"{cmidrule}")
    lines.append(f"\\textbf{{Model}} & {sub_header} \\\\")
    lines.append(r"\midrule")

    for base_name, post_name, _, _, _, _ in _MODELS:
        cells_b_all, cells_p_all = [], []
        for state_label in state_order:
            rb = all_results[state_label].get(base_name)
            rp = all_results[state_label].get(post_name)
            cb, cp = row_cells(rb, rp, state_label)
            cells_b_all.extend(cb)
            cells_p_all.extend(cp)
        lines.append(f"{base_name:<20s} & {' & '.join(cells_b_all)} \\\\")
        lines.append(f"$\\quad$+PoST       & {' & '.join(cells_p_all)} \\\\")
        lines.append(r"\midrule")

    # remove last \midrule before \bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    print("\n" + "=" * 70)
    print("LATEX TABLE:")
    print("=" * 70)
    print("\n".join(lines))
    print("=" * 70)

    out_path = os.path.join(SCRIPT_DIR, "mqar_table_output.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cached", action="store_true")
    args = parser.parse_args()

    if args.use_cached and os.path.exists(CACHE_FILE):
        print(f"Loading cache from {CACHE_FILE}")
        with open(CACHE_FILE) as f:
            all_data = json.load(f)
    else:
        print("Fetching from wandb ...")
        all_data = fetch_wandb_data()
        with open(CACHE_FILE, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"Cached to {CACHE_FILE}")

    print("\nGenerating table ...")
    generate_table(all_data)


if __name__ == "__main__":
    main()
