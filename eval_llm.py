"""
Zero-shot LM Evaluation for PoST Models.

Uses EleutherAI's lm-evaluation-harness to run standard benchmarks
and RULER NIAH retrieval tasks on pre-trained PoST models.

Supports multi-GPU data-parallel evaluation via accelerate.

Usage:
    # Single-GPU evaluation with default benchmarks:
    python eval_llm.py --model_path output/post_180m_10b/mamba2/mamba2

    # Multi-GPU evaluation (4 GPUs):
    accelerate launch --multi_gpu --num_processes=4 \\
        eval_llm.py --model_path output/post_180m_10b/mamba2/mamba2

    # Compare PoST vs baseline:
    python eval_llm.py \\
        --model_path output/post_180m_10b/mamba2_post/mamba2_post \\
        --model_path_baseline output/post_180m_10b/mamba2/mamba2

    # Run specific tasks only:
    python eval_llm.py --model_path ./checkpoint --tasks lambada_openai hellaswag

    # Standard benchmarks only (no NIAH):
    python eval_llm.py --model_path ./checkpoint --no_niah

    # Custom batch size (increase for faster eval on H200):
    python eval_llm.py --model_path ./checkpoint --batch_size 64
"""

import argparse
import json
import os
import sys
import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

# ==========================================
# Register custom model classes with Auto*
# ==========================================
# This lets HFLM load our models via string path, enabling
# full accelerate distributed setup (data-parallel multi-GPU).

from models import Mamba2PoSTForCausalLM, Mamba2PoSTConfig
from models import RWKV7PoSTForCausalLM, RWKV7Config
from models.post_gated_deltanet import GDNPoSTForCausalLM, GatedDeltaNetConfig

AutoConfig.register("mamba2_post", Mamba2PoSTConfig)
AutoModelForCausalLM.register(Mamba2PoSTConfig, Mamba2PoSTForCausalLM)

AutoConfig.register("rwkv7_post", RWKV7Config)
AutoModelForCausalLM.register(RWKV7Config, RWKV7PoSTForCausalLM)

AutoConfig.register("post_gated_deltanet", GatedDeltaNetConfig)
AutoModelForCausalLM.register(GatedDeltaNetConfig, GDNPoSTForCausalLM)


# ==========================================
# Default Task Lists
# ==========================================

# Standard LM benchmarks (zero-shot)
STANDARD_TASKS = [
    "lambada_openai",   # LAMB. ppl↓ / acc↑
    "hellaswag",        # HellaS. acc_n↑
    "piqa",             # PIQA acc↑
    "arc_easy",         # Arc-E acc↑
    "arc_challenge",    # Arc-C acc_n↑
    "winogrande",       # WinoGr. acc↑
    "openbookqa",       # OBQA acc↑
]

# RULER tasks (custom YAMLs in tasks/ with tokenizer metadata)
RULER_TASKS = [
    "post_ruler_cwe",
    "post_ruler_fwe",
    "post_ruler_vt",
    "post_ruler_qa_squad",
    "post_ruler_qa_hotpot",
]

# Default NIAH/RULER context lengths
DEFAULT_NIAH_LENGTHS = [2048, 4096, 8192]

# Default tokenizer (must match what was used for training)
DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B"


# ==========================================
# Evaluation
# ==========================================

def evaluate_model(model_path, tasks, batch_size, num_fewshot=0,
                   metadata=None, limit=None, include_path=None):
    """
    Load a model and evaluate it on the given tasks.

    Passes model_path as a string to HFLM, which uses
    AutoModelForCausalLM.from_pretrained() internally.
    This enables HFLM's full accelerate data-parallel setup
    when launched with `accelerate launch --multi_gpu`.

    Returns:
        results: dict from lm_eval with per-task metrics.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    print(f"\n📥 Loading model from {model_path}")

    # Pass model_path as string — HFLM handles loading + distributed setup
    lm = HFLM(
        pretrained=model_path,
        tokenizer=DEFAULT_TOKENIZER,
        batch_size=batch_size,
        dtype="bfloat16",
    )

    if lm.rank == 0:
        param_count = sum(p.numel() for p in lm.model.parameters()) / 1e6
        print(f"   🧠 {param_count:.1f}M params")
        print(f"\n🏃 Running evaluation: {', '.join(tasks)}")
        print(f"   Batch size: {batch_size} | Few-shot: {num_fewshot}")
        if lm.world_size > 1:
            print(f"   🚀 Data-parallel across {lm.world_size} GPUs")

    # NIAH/RULER tasks need a tokenizer name in metadata for data generation
    if metadata is None:
        metadata = {}
    if "tokenizer" not in metadata:
        metadata["tokenizer"] = DEFAULT_TOKENIZER

    # Register custom tasks from include_path
    task_manager = None
    if include_path:
        task_manager = TaskManager(include_path=include_path)
        if lm.rank == 0:
            print(f"   📂 Custom tasks from: {include_path}")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        metadata=metadata,
        limit=limit,
        task_manager=task_manager,
    )

    return results, lm.rank


def print_results_table(results, model_name="Model"):
    """Pretty-print evaluation results."""
    print(f"\n{'='*70}")
    print(f"  Results: {model_name}")
    print(f"{'='*70}")

    if "results" not in results:
        print("  No results found.")
        return

    # Separate standard tasks and NIAH tasks
    standard_results = {}
    niah_results = {}

    for task_name, task_results in results["results"].items():
        if "niah" in task_name.lower():
            niah_results[task_name] = task_results
        else:
            standard_results[task_name] = task_results

    # Print standard benchmarks
    if standard_results:
        print(f"\n  📊 Standard Benchmarks (zero-shot)")
        print(f"  {'Task':<25} {'Metric':<15} {'Score':>10}")
        print(f"  {'-'*50}")
        for task_name, task_results in standard_results.items():
            for metric, value in task_results.items():
                if metric.endswith(",none"):
                    metric_clean = metric.replace(",none", "")
                elif metric.endswith("_stderr"):
                    continue
                else:
                    metric_clean = metric

                if isinstance(value, float):
                    stderr_key = f"{metric}_stderr,none"
                    stderr = task_results.get(stderr_key, None)
                    if stderr and isinstance(stderr, float):
                        print(f"  {task_name:<25} {metric_clean:<15} {value:>8.4f} ± {stderr:.4f}")
                    else:
                        print(f"  {task_name:<25} {metric_clean:<15} {value:>8.4f}")

    # Print NIAH results
    if niah_results:
        print(f"\n  🔑 NIAH Retrieval (accuracy)")
        print(f"  {'Task':<20} ", end="")
        # Collect all context lengths
        all_lengths = set()
        for task_results in niah_results.values():
            for metric in task_results:
                if metric.endswith(",none"):
                    length_str = metric.replace(",none", "")
                    if length_str.isdigit():
                        all_lengths.add(int(length_str))
        all_lengths = sorted(all_lengths)

        for length in all_lengths:
            label = f"{length//1024}K" if length >= 1024 else str(length)
            print(f"{label:>8}", end="")
        print()
        print(f"  {'-'*20} " + "-" * (8 * len(all_lengths)))

        for task_name, task_results in niah_results.items():
            print(f"  {task_name:<20} ", end="")
            for length in all_lengths:
                key = f"{length},none"
                value = task_results.get(key, -1)
                if isinstance(value, float) and value >= 0:
                    print(f"{value:>7.1%}", end=" ")
                else:
                    print(f"{'N/A':>8}", end="")
            print()

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot LM Evaluation for PoST Models (Multi-GPU supported)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Multi-GPU Usage:
  accelerate launch --multi_gpu --num_processes=4 \\
      eval_llm.py --model_path ./checkpoint

  This uses data-parallel evaluation — each GPU gets a copy of the model
  and processes different batches. Speeds up evaluation ~linearly.
        """,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to PoST model checkpoint")
    parser.add_argument("--model_path_baseline", type=str, default=None,
                        help="Optional: path to baseline model for comparison")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Override evaluation tasks (default: standard benchmarks + RULER)")
    parser.add_argument("--no_ruler", action="store_true",
                        help="Skip RULER retrieval tasks")
    parser.add_argument("--ruler_lengths", nargs="+", type=int,
                        default=DEFAULT_NIAH_LENGTHS,
                        help="RULER context lengths (default: 2048 4096 8192)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Number of few-shot examples (0 = zero-shot)")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Directory to save results JSON")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per task (e.g. 30 for quick runs)")
    args = parser.parse_args()

    # Reproducibility
    set_seed(42)

    # Build task list
    if args.tasks is not None:
        tasks = args.tasks
    else:
        tasks = list(STANDARD_TASKS)
        if not args.no_ruler:
            tasks.extend(RULER_TASKS)

    # Metadata for RULER tasks (tokenizer + context lengths)
    metadata = {
        "tokenizer": DEFAULT_TOKENIZER,
        "max_seq_lengths": args.ruler_lengths,
    }

    # Custom task path for our NIAH overrides
    tasks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")
    include_path = tasks_dir if os.path.isdir(tasks_dir) else None

    # Evaluate main model
    results, rank = evaluate_model(
        args.model_path, tasks, args.batch_size, args.num_fewshot,
        metadata=metadata, limit=args.limit, include_path=include_path,
    )
    model_name = os.path.basename(os.path.normpath(args.model_path))

    # Only rank 0 prints and saves
    if rank == 0:
        print_results_table(results, model_name=model_name)

        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"lm_eval_{model_name}.json")
        with open(out_path, "w") as f:
            json.dump(results.get("results", {}), f, indent=2, default=str)
        print(f"💾 Results saved to {out_path}")

    # Evaluate baseline if provided
    if args.model_path_baseline:
        results_bl, rank_bl = evaluate_model(
            args.model_path_baseline, tasks, args.batch_size,
            args.num_fewshot, metadata=metadata, limit=args.limit,
            include_path=include_path,
        )
        baseline_name = os.path.basename(os.path.normpath(args.model_path_baseline))

        if rank_bl == 0:
            print_results_table(results_bl, model_name=baseline_name)

            out_path_bl = os.path.join(args.output_dir, f"lm_eval_{baseline_name}.json")
            with open(out_path_bl, "w") as f:
                json.dump(results_bl.get("results", {}), f, indent=2, default=str)
            print(f"💾 Baseline results saved to {out_path_bl}")


if __name__ == "__main__":
    main()
