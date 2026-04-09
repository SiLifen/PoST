"""
Multi-Architecture Trainer for PoST Project
=============================================
Trains 0.5B-class language models across four architectures:
  - mamba2       : Vanilla Mamba-2
  - mamba2_post  : Mamba-2 with PoST (Position-Adaptive Δ-Scaling)
  - rwkv7        : Vanilla RWKV-7
  - rwkv7_post   : RWKV-7 with PoST (Position-Adaptive Decay)

Usage:
    # Train all 4 architectures sequentially
    python trainer.py

    # Train a single architecture
    python trainer.py --arch mamba2_post

    # Dry run — print param counts only
    python trainer.py --dry_run

    # Multi-GPU (recommended)
    accelerate launch --multi_gpu --num_processes=8 trainer.py
    # or:  torchrun --nproc_per_node=8 trainer.py
"""

import os
import json
import argparse
import math
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from itertools import chain

from models import (
    Mamba2PoSTConfig,
    Mamba2PoSTForCausalLM,
    RWKV7Config,
    RWKV7PoSTForCausalLM,
)
from models.post_gated_deltanet import (
    GatedDeltaNetConfig,
    GDNPoSTForCausalLM,
)


# ==========================================
# Architecture Registry
# ==========================================

ALL_ARCHS = ["mamba2", "mamba2_post", "rwkv7", "rwkv7_post", "gdn", "gdn_post"]
ALL_SIZES = ["180M", "440M", "880M", "1.5B"]

ARCH_DISPLAY_NAMES = {
    "mamba2":      "Mamba-2 (Baseline)",
    "mamba2_post": "Mamba-2 PoST",
    "rwkv7":       "RWKV-7 (Baseline)",
    "rwkv7_post":  "RWKV-7 PoST",
    "gdn":         "Gated DeltaNet (Baseline)",
    "gdn_post":    "Gated DeltaNet PoST",
}

MODEL_CONFIGS = {
    "mamba2": {
        "180M": {"d_model": 768,  "n_layer": 24},  
        "440M": {"d_model": 1024, "n_layer": 48}, 
        "880M": {"d_model": 1536, "n_layer": 48},  
        "1.5B":   {"d_model": 2048, "n_layer": 48},
    },
    "rwkv7": {
        "180M": {"d_model": 768,  "n_layer": 12}, 
        "440M": {"d_model": 1024, "n_layer": 24},  
        "880M": {"d_model": 1536, "n_layer": 24},  
        "1.5B":   {"d_model": 2048, "n_layer": 24}, 
    },
    "gdn": {
        "180M": {"d_model": 768,  "n_layer": 24, "num_heads": 6},
        "440M": {"d_model": 1024, "n_layer": 48, "num_heads": 8},
        "880M": {"d_model": 1536, "n_layer": 48, "num_heads": 12},
        "1.5B":   {"d_model": 2048, "n_layer": 48, "num_heads": 16},
    },
}


def create_model(arch: str, vocab_size: int, max_seq_length: int, model_size: str = "400M"):
    """
    Create a model and config for the given architecture and size.

    Returns:
        (model, config)
    """
    # Look up the base arch key for config
    if "mamba2" in arch:
        arch_key = "mamba2"
    elif "rwkv" in arch:
        arch_key = "rwkv7"
    elif "gdn" in arch:
        arch_key = "gdn"
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    size_cfg = MODEL_CONFIGS[arch_key][model_size]
    d_model = size_cfg["d_model"]
    n_layer = size_cfg["n_layer"]

    if arch in ("mamba2", "mamba2_post"):
        config = Mamba2PoSTConfig(
            hidden_size=d_model,
            num_hidden_layers=n_layer,
            state_size=128,
            conv_kernel=4,
            expand=2,
            head_dim=64,
            chunk_size=256,
            vocab_size=vocab_size,
            tie_word_embeddings=True,
            train_length=max_seq_length,
            use_post=("post" in arch),
        )
        model = Mamba2PoSTForCausalLM(config)
        return model, config

    elif arch in ("rwkv7", "rwkv7_post"):
        config = RWKV7Config(
            d_model=d_model,
            n_layer=n_layer,
            head_dim=64,
            vocab_size=vocab_size,
            pad_vocab_size_multiple=16,
            tie_embeddings=True,
            train_length=max_seq_length,
            use_post=("post" in arch),
        )
        model = RWKV7PoSTForCausalLM(config)
        return model, config

    elif arch in ("gdn", "gdn_post"):
        num_heads = size_cfg.get("num_heads", 6)
        config = GatedDeltaNetConfig(
            d_model=d_model,
            n_layer=n_layer,
            num_heads=num_heads,
            expand_v=2,
            use_gate=True,
            use_short_conv=True,
            conv_size=4,
            vocab_size=vocab_size,
            pad_vocab_size_multiple=16,
            tie_embeddings=True,
            train_length=max_seq_length,
            use_post=("post" in arch),
        )
        model = GDNPoSTForCausalLM(config)
        return model, config

    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {ALL_ARCHS}")


def count_unique_params(model) -> int:
    """Count unique parameters (respects weight tying)."""
    seen_ptrs = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen_ptrs:
            seen_ptrs.add(p.data_ptr())
            total += p.numel()
    return total


# ==========================================
# Data Processing
# ==========================================

def _detect_text_column(dataset):
    """Auto-detect the text column in a dataset."""
    for col in ["text", "content", "passage", "document"]:
        if col in dataset.column_names:
            return col
    raise ValueError(
        f"Could not find text column. Available: {dataset.column_names}"
    )


def prepare_dataset(dataset_name, tokenizer, max_seq_length, no_packing=False, text_column=None, dataset_config=None, max_tokens=None, data_path=None, prepare_mode=False):
    """
    Load dataset and tokenize.

    If data_path is provided, loads a pre-tokenized and pre-packed dataset
    directly from disk, bypassing Hugging Face downloads entirely.
    (Unless prepare_mode=True, in which case it will process the data and save it to data_path).

    If no_packing=False (default): concatenate all texts and split into
        fixed-length chunks (data packing).
    If no_packing=True: tokenize each document independently, truncate
        to max_seq_length. No cross-document contamination.
    """
    print(f"DEBUG: prepare_mode={prepare_mode}, data_path={data_path}, os.path.exists={os.path.exists(data_path) if data_path else False}")
    
    if data_path and os.path.exists(data_path) and not prepare_mode:
        print(f"   Using existing local packaged dataset: {data_path}")
        dataset = load_from_disk(data_path)
        return dataset
    elif prepare_mode and data_path and os.path.exists(data_path):
        # We are trying to prepare data into data_path, but it already exists. We can just load it.
        print(f"   Dataset already exists at {data_path}. Loading existing.")
        dataset = load_from_disk(data_path)
        return dataset
    else:
        try:
            dataset = load_dataset(dataset_name, name=dataset_config, split="train")
        except Exception as e:
            print(f"⚠️  load_dataset failed: {e}")
            print("   If you have already processed and cached the dataset, make sure you pass --data_path.")
            import sys
            sys.exit(1)

    if text_column is None:
        text_column = _detect_text_column(dataset)

    print(f"   Text column: '{text_column}'")
    print(f"   Raw examples: {len(dataset):,}")

    if max_tokens is not None:
        print(f"   Estimating examples needed to reach ~{max_tokens/1e9:.2f}B tokens...")
        sample_size = min(1000, len(dataset))
        sample = dataset.select(range(sample_size))
        sample_tokens = sum(len(tokenizer(x, truncation=False, add_special_tokens=False)["input_ids"]) for x in sample[text_column])
        avg_tokens = sample_tokens / sample_size
        examples_needed = int((max_tokens / avg_tokens) * 1.05)  # 5% buffer
        if examples_needed < len(dataset):
            print(f"   Subsetting dataset: taking {examples_needed:,} examples (avg {avg_tokens:.0f} tokens/doc)")
            dataset = dataset.select(range(examples_needed))

    if no_packing:
        print(f"📄 Tokenizing per-document (no packing), max_length={max_seq_length}...")

        def tokenize_fn(examples):
            tokenized = tokenizer(
                examples[text_column],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                add_special_tokens=True,
            )
            # Mask padding tokens in labels with -100 so they're ignored in loss
            labels = []
            for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
                labels.append([id_ if m == 1 else -100 for id_, m in zip(ids, mask)])
            tokenized["labels"] = labels
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=2000,
            num_proc=os.cpu_count(),
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

        avg_len = sum(len(x["input_ids"]) for x in tokenized_dataset.select(range(min(1000, len(tokenized_dataset))))) / min(1000, len(tokenized_dataset))
        print(f"   Sequences: {len(tokenized_dataset):,} (avg {avg_len:.0f} tokens)")

    else:
        print(f"📦 Tokenizing & Packing to length {max_seq_length}...")

        def group_texts(examples):
            tokenized = tokenizer(
                examples[text_column],
                truncation=False,
                add_special_tokens=False,
            )
            concatenated = {
                k: list(chain(*tokenized[k]))
                for k in tokenized.keys()
            }
            total_length = len(concatenated[list(concatenated.keys())[0]])
            total_length = (total_length // max_seq_length) * max_seq_length
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_dataset = dataset.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=os.cpu_count(),
            remove_columns=dataset.column_names,
            desc="Packing",
        )

        total_tokens = len(tokenized_dataset) * max_seq_length
        print(f"   Packed sequences: {len(tokenized_dataset):,}")
        print(f"   Total tokens: {total_tokens / 1e9:.2f}B")

    return tokenized_dataset


class StreamingPackedDataset(IterableDataset):
    """
    IterableDataset that streams from HuggingFace, tokenizes on-the-fly,
    and packs tokens into fixed-length sequences. No disk caching.
    """

    def __init__(self, dataset_name, dataset_config, tokenizer, max_seq_length,
                 text_column=None, seed=42):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_column = text_column
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # Each worker and each DDP rank gets a different shard
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        # Combine DDP rank and dataloader worker for unique sharding
        if worker_info is not None:
            total_workers = world_size * worker_info.num_workers
            global_worker_id = rank * worker_info.num_workers + worker_info.id
        else:
            total_workers = world_size
            global_worker_id = rank

        ds = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split="train",
            streaming=True,
        )

        # Deterministic shuffling + sharding
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)

        # Detect text column from first example if needed
        text_col = self.text_column
        if text_col is None:
            # Peek at features
            for col in ["text", "content", "passage", "document"]:
                if col in ds.column_names:
                    text_col = col
                    break
            if text_col is None:
                raise ValueError(f"Cannot detect text column from: {ds.column_names}")

        # Buffer for packing
        token_buffer = []
        sample_idx = 0

        for example in ds:
            # Manual sharding: each worker only processes its share
            if sample_idx % total_workers != global_worker_id:
                sample_idx += 1
                continue
            sample_idx += 1

            text = example[text_col]
            if not text or not text.strip():
                continue

            tokens = self.tokenizer(
                text, truncation=False, add_special_tokens=False,
            )["input_ids"]
            token_buffer.extend(tokens)

            # Yield complete chunks
            while len(token_buffer) >= self.max_seq_length:
                chunk = token_buffer[:self.max_seq_length]
                token_buffer = token_buffer[self.max_seq_length:]
                yield {
                    "input_ids": chunk,
                    "attention_mask": [1] * self.max_seq_length,
                    "labels": chunk,
                }


def prepare_streaming_dataset(dataset_name, tokenizer, max_seq_length,
                              dataset_config=None, text_column=None, seed=42):
    """
    Create a streaming packed dataset — no disk caching.
    Returns an IterableDataset.
    """
    print(f"\n🌊 Streaming Dataset: {dataset_name}"
          f"{f' ({dataset_config})' if dataset_config else ''}")
    print(f"   Packing to length {max_seq_length}, no disk cache")

    return StreamingPackedDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        text_column=text_column,
        seed=seed,
    )


# ==========================================
# Periodic LM-Eval Callback
# ==========================================

class LMEvalCallback(TrainerCallback):
    """
    TrainerCallback that periodically runs lm-eval benchmarks during training.

    Runs lightweight evaluation (e.g., lambada_openai) every `eval_steps`
    global steps to monitor model quality and catch bad runs early.
    Only runs on the main process (rank 0) in multi-GPU setups.

    Results are logged to wandb and saved as JSON files.
    """

    def __init__(self, tokenizer, eval_tasks, eval_steps, eval_batch_size=32, output_dir="output"):
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_tasks = eval_tasks
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.output_dir = output_dir
        self._last_eval_step = -1

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Check if we should run eval at this step."""
        if self.eval_steps <= 0:
            return
        if state.global_step == 0:
            return
        if state.global_step == self._last_eval_step:
            return
        if state.global_step % self.eval_steps != 0:
            return

        # Only run on main process
        if not state.is_world_process_zero:
            return

        self._run_eval(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Run final evaluation when training ends."""
        if not state.is_world_process_zero:
            return
        if state.global_step != self._last_eval_step:
            self._run_eval(model, state.global_step, final=True)

    @torch.no_grad()
    def _run_eval(self, model, global_step, final=False):
        """Run lm-eval and log results."""
        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
        except ImportError:
            print("⚠️  lm-eval not installed, skipping periodic evaluation. "
                  "Install with: pip install lm-eval")
            return

        tag = "final" if final else f"step-{global_step}"
        print(f"\n📊 LM-Eval @ {tag}: running {', '.join(self.eval_tasks)}...")

        self._last_eval_step = global_step
        was_training = model.training
        model.eval()

        try:
            lm = HFLM(
                pretrained=model,
                tokenizer=self.tokenizer,
                batch_size=self.eval_batch_size,
            )

            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=self.eval_tasks,
                num_fewshot=0,
                batch_size=self.eval_batch_size,
            )

            # Extract and log metrics
            metrics = {}
            if "results" in results:
                for task_name, task_results in results["results"].items():
                    for metric_key, value in task_results.items():
                        if isinstance(value, (int, float)) and "stderr" not in metric_key:
                            clean_key = metric_key.replace(",none", "")
                            metrics[f"lm_eval/{task_name}/{clean_key}"] = value

            # Print summary
            print(f"   📈 LM-Eval Results @ {tag}:")
            for k, v in sorted(metrics.items()):
                print(f"      {k}: {v:.4f}")

            # Log to wandb
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({**metrics, "train/global_step": global_step})
            except ImportError:
                pass

            # Save to JSON
            eval_dir = os.path.join(self.output_dir, "eval_results")
            os.makedirs(eval_dir, exist_ok=True)
            eval_path = os.path.join(eval_dir, f"lm_eval_{tag}.json")
            with open(eval_path, "w") as f:
                json.dump({"global_step": global_step, "metrics": metrics}, f, indent=2)
            print(f"   💾 Saved to {eval_path}")

        except Exception as e:
            print(f"   ❌ LM-Eval failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if was_training:
                model.train()


# ==========================================
# Selective Weight Decay Parameter Groups
# ==========================================

def build_optimizer_param_groups(model, weight_decay: float, base_lr: float):
    """
    Build AdamW parameter groups with selective weight decay.

    Applies the official RWKV-7 / GPT-style rule universally across all
    architectures (Mamba-2, RWKV-7, GDN, …):

      lr_2x (no WD): "w0_base" or "w0_delta" in name
                     → PoST cumulative-softplus decay params (2× LR)
      lr_decay (WD): p.squeeze().ndim >= 2  AND  ".weight" in name
                     → full projection matrices, LoRA matrices
      lr_1x (no WD): everything else
                     → biases, LN/GN params, 1-D scalars (A_log, D, k_k, …)

    The dimension heuristic naturally excludes Mamba's A_log + D (1-D),
    all biases, and LayerNorm/GroupNorm params without any model-side flags.

    Parameters marked with ``_lr_scale`` (e.g. RWKV-7 PoST's w0_base/w0_delta
    have ``_lr_scale = 2.0``) are grouped by that scale and assigned no WD.
    """
    lr_decay  = {}            # name → param  (WD, 1× LR)
    lr_scaled = {}            # name → (param, scale)  (no WD, n× LR)
    lr_1x    = {}             # name → param  (no WD, 1× LR)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_scale = getattr(param, '_lr_scale', 1.0)
        if lr_scale != 1.0:
            lr_scaled[name] = (param, lr_scale)
        elif (param.squeeze().dim() >= 2) and (".weight" in name) and (weight_decay > 0):
            lr_decay[name] = param
        else:
            lr_1x[name] = param

    wd_numel  = sum(p.numel() for p in lr_decay.values())
    sc_numel  = sum(p.numel() for p, _ in lr_scaled.values())
    x1_numel  = sum(p.numel() for p in lr_1x.values())
    print(f"   Param groups  |  "
          f"WD ({weight_decay}): {wd_numel/1e6:.2f}M  |  "
          f"scaled-LR no-WD: {sc_numel/1e6:.2f}M  |  "
          f"1×LR no-WD: {x1_numel/1e6:.2f}M")
    if lr_scaled:
        print(f"   Scaled-LR params: "
              f"{ {n: s for n, (_, s) in lr_scaled.items()} }")

    # Build groups: one per unique lr_scale value for scaled params
    optim_groups = [
        {"params": list(lr_1x.values()), "weight_decay": 0.0, "lr": base_lr},
    ]
    # Collect by scale
    from collections import defaultdict
    by_scale = defaultdict(list)
    for param, scale in lr_scaled.values():
        by_scale[scale].append(param)
    for scale, params in by_scale.items():
        optim_groups.append({"params": params, "weight_decay": 0.0, "lr": base_lr * scale})
    if weight_decay > 0 and lr_decay:
        optim_groups.append(
            {"params": list(lr_decay.values()), "weight_decay": weight_decay, "lr": base_lr}
        )
    return optim_groups


class SelectiveWDTrainer(Trainer):
    """
    HuggingFace Trainer that uses build_optimizer_param_groups() for all
    architectures, applying the standard GPT/RWKV weight-decay heuristic:
    only >=2D weight matrices receive WD; everything else (biases, 1-D
    scalars, LN/GN params) does not.
    """

    def __init__(self, weight_decay: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._wd = weight_decay

    def create_optimizer(self):
        param_groups = build_optimizer_param_groups(
            self.model, self._wd, self.args.learning_rate
        )
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        optimizer_kwargs.pop("weight_decay", None)  # WD already in param groups
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

# Keep old name as alias for backwards compatibility
RWKV7Trainer = SelectiveWDTrainer

# ==========================================
# Training
# ==========================================

def train_single_arch(
    arch: str,
    tokenized_dataset,
    tokenizer,
    args,
    use_bf16: bool,
):
    """Train a single architecture from scratch."""
    # Fix seed BEFORE model creation for reproducible weight init
    set_seed(42)

    display_name = ARCH_DISPLAY_NAMES[arch]
    print(f"\n{'='*60}")
    print(f"🚀 Training: {display_name}")
    print(f"{'='*60}")

    # Create model
    model, config = create_model(arch, len(tokenizer), args.max_seq_length, args.model_size)
    num_params = count_unique_params(model)
    print(f"🧠 Model: {num_params / 1e6:.1f}M params (unique)")

    # Output directory
    output_dir = os.path.join(args.output_dir, arch)
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments — adam_betas and eps differ by architecture; WD is
    # handled per-param-group by SelectiveWDTrainer (set to 0.0 here so the
    # HF Trainer global WD doesn't override the per-group values).
    is_rwkv = "rwkv" in arch
    use_max_steps = hasattr(args, 'max_steps') and args.max_steps is not None and args.max_steps > 0
    training_args = TrainingArguments(
        run_name=f"post-{arch}-{args.model_size}",
        output_dir=output_dir,
        num_train_epochs=args.epochs if not use_max_steps else 100,
        max_steps=args.max_steps if use_max_steps else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.0,          # SelectiveWDTrainer handles WD via param groups
        adam_beta1=0.9,
        adam_beta2=0.99 if is_rwkv else 0.95,
        adam_epsilon=1e-18 if is_rwkv else 1e-8,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-5},
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=use_bf16,
        tf32=True,
        max_grad_norm=1.0,
        report_to="wandb",
        push_to_hub=False,
        dataloader_num_workers=4,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if use_max_steps:
        print(f"\u23f1\ufe0f  Training for max {args.max_steps} steps")

    # Callbacks
    callbacks = []
    if args.eval_steps > 0:
        callbacks.append(
            LMEvalCallback(
                tokenizer=tokenizer,
                eval_tasks=args.eval_tasks,
                eval_steps=args.eval_steps,
                eval_batch_size=args.eval_batch_size,
                output_dir=output_dir,
            )
        )
        print(f"\U0001f4ca LM-Eval enabled: every {args.eval_steps} steps, "
              f"tasks={args.eval_tasks}")

    # All architectures use SelectiveWDTrainer for correct per-group WD.
    trainer = SelectiveWDTrainer(
        weight_decay=0.1,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks,
    )

    print(f"🔥 Starting training...")
    # Auto-detect checkpoint for resume
    resume_ckpt = None
    if args.resume:
        # Find latest checkpoint-* dir
        ckpts = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        ) if os.path.exists(output_dir) else []
        if ckpts:
            resume_ckpt = os.path.join(output_dir, ckpts[-1])
            print(f"📂 Resuming from {resume_ckpt}")
        else:
            print("⚠️  --resume set but no checkpoint found, starting fresh.")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Done: {display_name} → {output_dir}")

    # Free GPU memory before next model
    del model, trainer
    torch.cuda.empty_cache()


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Train PoST models across multiple architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all 4 architectures
  python trainer.py --dataset_name cerebras/SlimPajama-627B

  # Train one architecture
  python trainer.py --arch mamba2_post

  # Dry run (just print param counts)
  python trainer.py --dry_run

  # Multi-GPU
  accelerate launch --multi_gpu --num_processes=8 trainer.py
        """,
    )

    # Architecture selection
    parser.add_argument(
        "--arch", type=str, default=None,
        choices=ALL_ARCHS,
        help="Train a single architecture. If not set, trains ALL four sequentially.",
    )
    parser.add_argument(
        "--model_size", type=str, default="180M",
        choices=ALL_SIZES,
        help="Model size preset (default: 180M)",
    )

    # Data
    parser.add_argument(
        "--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset name (default: FineWeb-Edu)",
    )
    parser.add_argument(
        "--dataset_config", type=str, default="sample-100BT",
        help="Dataset config/subset name (default: sample-100BT)",
    )
    parser.add_argument(
        "--max_tokens", type=float, default=None,
        help="Max tokens to train on (e.g. 10e9 for 10B). "
             "If set, overrides --epochs and computes the equivalent epoch fraction.",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to a pre-tokenized local dataset directory (e.g. from save_to_disk). "
             "If provided, bypasses Hugging Face downloads entirely.",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Context length")
    parser.add_argument("--packing", action="store_true",
                        help="Enable data packing (concatenate documents). Default: off")
    parser.add_argument("--streaming", action="store_true",
                        help="Stream dataset from HuggingFace (no disk caching). "
                             "Requires --max_tokens. Implies --packing.")
    parser.add_argument("--prepare_data_only", action="store_true",
                        help="If set, only downloads, tokenizes, and (if packing enabled) packs "
                             "the dataset, saves it to --data_path, and exits without training.")

    # Training hyperparams
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of epochs (ignored if --max_steps is set)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Train for exactly N steps then stop. Enables pause/resume workflow. "
                             "Use with --resume to continue from checkpoint.")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--output_dir", type=str, default="output", help="Base output directory")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing (saves VRAM)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output dir")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps (default: 1000)")

    # Periodic LM-Eval
    parser.add_argument(
        "--eval_steps", type=int, default=2000,
        help="Run lm-eval every N steps (0 to disable). Default: 2000",
    )
    parser.add_argument(
        "--eval_tasks", nargs="+",
        default=["lambada_openai", "hellaswag"],
        help="LM-eval tasks to run periodically (default: lambada_openai hellaswag)",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32,
        help="Batch size for lm-eval (default: 32)",
    )

    # Utilities
    parser.add_argument("--dry_run", action="store_true", help="Only create models and print param counts, no training")

    args = parser.parse_args()

    # ---- Hardware Setup ----
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        num_gpus = torch.cuda.device_count()
        print(f"🖥️  GPUs: {num_gpus}× {gpu_name}")
        use_bf16 = torch.cuda.is_bf16_supported()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("⚠️  No CUDA — training will be extremely slow.")
        use_bf16 = False

    # ---- Tokenizer ----
    tokenizer_name = "meta-llama/Llama-3.1-8B"
    print(f"📚 Tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Suppress tokenizer maximum sequence length warnings during data packing
    tokenizer.model_max_length = int(1e9)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Which architectures to train ----
    archs_to_train = [args.arch] if args.arch else ALL_ARCHS

    # ---- Dry Run: print param counts and exit ----
    if args.dry_run:
        print(f"\n{'='*60}")
        print("🔍 DRY RUN — Model Parameter Counts")
        print(f"{'='*60}")
        for arch in archs_to_train:
            model, config = create_model(arch, len(tokenizer), args.max_seq_length, args.model_size)
            num_params = count_unique_params(model)
            print(f"  {ARCH_DISPLAY_NAMES[arch]:30s} ({args.model_size}) → {num_params / 1e6:>8.1f}M params")
            del model
        print()
        return

    # ---- Prepare Dataset (once, shared across all models) ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.streaming:
        # ── Streaming mode: no disk cache ──
        if args.max_tokens is None:
            parser.error("--streaming requires --max_tokens to compute max_steps")

        tokens_per_step = (
            args.batch_size * args.grad_accum * world_size * args.max_seq_length
        )
        args.max_steps = int(args.max_tokens / tokens_per_step)
        print(f"🌊 Streaming mode: {args.max_tokens/1e9:.0f}B tokens "
              f"/ {tokens_per_step:,} tokens/step = {args.max_steps:,} steps")

        tokenized_dataset = prepare_streaming_dataset(
            args.dataset_name, tokenizer, args.max_seq_length,
            dataset_config=args.dataset_config,
        )
    else:
        # ── Standard mode: full download + cache to disk ──
        # In multi-GPU, only rank 0 does the expensive tokenization/packing;
        # other ranks wait at a barrier and then load the cached result.
        if world_size > 1:
            import torch.distributed as dist
            import datetime
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))

            if local_rank == 0:
                tokenized_dataset = prepare_dataset(
                    args.dataset_name, tokenizer, args.max_seq_length,
                    no_packing=not args.packing,
                    dataset_config=args.dataset_config,
                    max_tokens=args.max_tokens,
                    data_path=args.data_path,
                )
            dist.barrier()  # all ranks wait for rank 0 to finish
            if local_rank != 0:
                tokenized_dataset = prepare_dataset(
                    args.dataset_name, tokenizer, args.max_seq_length,
                    no_packing=not args.packing,
                    dataset_config=args.dataset_config,
                    max_tokens=args.max_tokens,
                    data_path=args.data_path,
                )
        else:
            tokenized_dataset = prepare_dataset(
                args.dataset_name, tokenizer, args.max_seq_length,
                no_packing=not args.packing,
                dataset_config=args.dataset_config,
                max_tokens=args.max_tokens,
                data_path=args.data_path,
                prepare_mode=args.prepare_data_only,
            )

        if args.prepare_data_only:
            if not args.data_path:
                print("❌ Error: --data_path is required when using --prepare_data_only to specify where to save the dataset.")
                return
            
            # In multi-GPU settings, only rank 0 should save to avoid race conditions.
            # But normally prepare_data_only should be run on a single machine.
            if local_rank == 0:
                print(f"\n💾 Saving processed dataset to '{args.data_path}'...")
                os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
                tokenized_dataset.save_to_disk(args.data_path, num_proc=min(16, os.cpu_count()))
                print("✅ Data preparation complete! Exiting.")
            
            if world_size > 1:
                import torch.distributed as dist
                dist.barrier()
            return

        # If --max_tokens is set, compute the epoch fraction
        total_tokens_in_dataset = len(tokenized_dataset) * args.max_seq_length
        if args.max_tokens is not None:
            desired_tokens = args.max_tokens
            args.epochs = desired_tokens / total_tokens_in_dataset
            print(f"📊 --max_tokens={desired_tokens/1e9:.1f}B → epochs={args.epochs:.4f}")
            if args.epochs > 1.0:
                print(f"   ⚠️ Dataset only has {total_tokens_in_dataset/1e9:.1f}B tokens, "
                      f"will train for {args.epochs:.2f} epochs.")

    # ---- Train Each Architecture ----
    print(f"\n🎯 Training plan: {len(archs_to_train)} model(s)")
    for i, arch in enumerate(archs_to_train, 1):
        print(f"\n[{i}/{len(archs_to_train)}] {ARCH_DISPLAY_NAMES[arch]}")

    for i, arch in enumerate(archs_to_train, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(archs_to_train)}] {ARCH_DISPLAY_NAMES[arch]}")
        print(f"{'#'*60}")

        train_single_arch(
            arch=arch,
            tokenized_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            args=args,
            use_bf16=use_bf16,
        )

    print(f"\n🏁 All training complete! Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
