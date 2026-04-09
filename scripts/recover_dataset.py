#!/usr/bin/env python3
"""
Recover a tokenized dataset from HuggingFace arrow cache files.

Usage:
    python scripts/recover_dataset.py \
        --hf_home /path/to/hf_cache \
        --cache_dir /path/to/arrow/cache \
        --save_path /path/to/output/dataset
"""
import argparse
import os
import glob
from datasets import config as ds_config
ds_config.IN_MEMORY_MAX_SIZE = 1000000000  # 1GB memory limit to force disk usage
from datasets import load_dataset, Features, Sequence, Value


def main():
    parser = argparse.ArgumentParser(description="Recover tokenized dataset from HF arrow cache.")
    parser.add_argument("--hf_home", type=str, required=True,
                        help="Path to HuggingFace cache directory")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Path to the arrow cache directory containing cache-*.arrow files")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Output path for the recovered dataset")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_home
    os.environ["HF_DATASETS_CACHE"] = args.hf_home
    import datasets
    datasets.config.HF_DATASETS_CACHE = args.hf_home

    print("Gathering cache files...")
    files = glob.glob(os.path.join(args.cache_dir, "cache-*.arrow"))

    if not files:
        print("No cache files found. Exiting.")
        return

    print(f"Found {len(files)} arrow cache files.")

    features = Features({
        'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
        'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
        'labels': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None)
    })

    print("Loading dataset from arrow chunks...")
    ds = load_dataset(
        "arrow",
        data_files={"train": files},
        split="train",
        features=features,
        cache_dir=args.hf_home
    )

    print(f"Loaded dataset with {len(ds)} packed sequences.")
    print(f"Saving to: {args.save_path} ...")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    num_proc = min(os.cpu_count(), 16)
    ds.save_to_disk(args.save_path, num_proc=num_proc)

    print(f"Successfully saved permanent dataset to {args.save_path}!")


if __name__ == "__main__":
    main()
