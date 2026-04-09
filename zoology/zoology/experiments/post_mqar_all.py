"""
PoST MQAR — Complete Reproducibility Script (All Architectures).

Runs ALL architectures on the MQAR task at three state sizes:
  - 64K : d=512, nheads=4   (2×512×64 = 512²/4  = 64K)  ← NEW
  - 32K : d=512, nheads=8   (2×512×32 = 512²/8  = 32K)  ← done
  - 16K : d=256, nheads=4   (2×256×32 = 256²/4  = 16K)  ← done

Models (all three state sizes):
  Mamba-2 + Mamba-2 PoST, RWKV-7 + RWKV-7 PoST,
  GDN + GDN PoST, GLA, RetNet, GLA PoST (= RetNet PoST)

d_state is derived automatically: d_state = d_model // (2 × nheads),
ensuring 2·d·d_state == d²/nheads across all architectures.

Usage:
    python -m zoology.launch zoology/experiments/post_mqar_all.py
"""

import uuid
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = f"post-mqar-all-{sweep_id}"
PROJECT_NAME = "post-mqar-state-equalized"

# Add run_ids here to skip already-completed runs
SKIP_RUNS = {

}

VOCAB_SIZE = 8_192
NUM_EXAMPLES_PER_STAGE = 2**18  # 262144
TOTAL_BATCH_TOKENS = 2**18      # 262144 tokens per batch

TRAIN_SEQ_LEN = 512
TEST_SEQ_LENS = [512, 1024, 2048, 4096]

# (d_model, nheads) pairs and state sizes:
#   d=512, nheads=8  →  state = 512²/8  = 32K  (already done)
#   d=256, nheads=4  →  state = 256²/4  = 16K  (already done)
#   d=512, nheads=4  →  state = 512²/4  = 64K  (new)
# d_state is computed per-run: d_state = d_model // (2 * nheads)
# so that  2 × d × d_state  ==  d² / nheads  (equalized across archs)
D_MODELS = [512, 256]

NHEADS_BY_DMODEL = {
    512: [4, 8],   # h=4 → 64K (new),  h=8 → 32K (done)
    256: [4],      # h=4 → 16K (done)
}

LR_SWEEP_RWKV  = [10**-4.5, 10**-4, 10**-3.5]
LR_SWEEP_MAMBA = {
    512: [10**-3, 10**-2.5, 10**-2],
    256: [10**-3, 10**-2.5, 10**-2],
}
LR_SWEEP_GDN = [10**-3.5, 10**-3, 10**-2.5]
LR_SWEEP_GLA_RETNET = [10**-3, 10**-2.5, 10**-2]

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
}

# ============================================================
# Data Configuration
# ============================================================

kv_schedule = [TRAIN_SEQ_LEN // 32, TRAIN_SEQ_LEN // 16,
               TRAIN_SEQ_LEN // 8, TRAIN_SEQ_LEN // 4]
train_configs = [
    MQARConfig(
        vocab_size=VOCAB_SIZE,
        input_seq_len=TRAIN_SEQ_LEN,
        num_examples=NUM_EXAMPLES_PER_STAGE,
        num_kv_pairs=kv,
        random_non_queries=True,
    )
    for kv in kv_schedule
]

FIXED_KV = TRAIN_SEQ_LEN // 4  # 128

persist_tests = [
    MQARConfig(
        vocab_size=VOCAB_SIZE,
        input_seq_len=test_T,
        num_examples=3_000,
        num_kv_pairs=FIXED_KV,
        random_non_queries=True,
        test_type="persist",
    )
    for test_T in TEST_SEQ_LENS
]

capacity_tests = [
    MQARConfig(
        vocab_size=VOCAB_SIZE,
        input_seq_len=test_T,
        num_examples=3_000,
        num_kv_pairs=test_T // 4,
        random_non_queries=True,
        test_type="capacity",
    )
    for test_T in TEST_SEQ_LENS if test_T > TRAIN_SEQ_LEN
]

test_configs = persist_tests + capacity_tests

batch_size = TOTAL_BATCH_TOKENS // TRAIN_SEQ_LEN
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(batch_size, max(batch_size // 8, 1)),
    cache_dir="/tmp/zoology_cache_mqar_all",
)

# ============================================================
# Build configs
# ============================================================
configs = []

for d_model in D_MODELS:
    expand = 2
    d_inner = d_model * expand
    lr_sweep_mamba = LR_SWEEP_MAMBA[d_model]

    for nheads in NHEADS_BY_DMODEL[d_model]:
        # equalized: 2·d·d_state = d²/nheads
        d_state = d_model // (2 * nheads)
        ssm_headdim = d_inner // nheads
        rwkv_headdim = d_model // nheads

        # ===========================================
        # Mamba-2 PoST
        # ===========================================
        post_mixer = dict(
            name="zoology.mixers.post_mamba2_mixer.Mamba2PoSTMixer",
            kwargs={
                "d_state": d_state,
                "headdim": ssm_headdim,
                "train_length": TRAIN_SEQ_LEN,
            },
        )
        post_model = ModelConfig(
            block_type="Mamba2PoSTBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=post_mixer,
            max_position_embeddings=0,
            name="mamba2_post",
            **model_factory_kwargs,
        )
        for lr in lr_sweep_mamba:
            rid = f"mamba2_post-h{nheads}-d{d_model}-lr{lr:.1e}"
            if rid in SKIP_RUNS:
                print(f"  ⏭  Skipping {rid} (already done)")
                continue
            configs.append(TrainConfig(
                model=post_model, data=data, learning_rate=lr,
                max_epochs=8, lr_scheduler_type="linear_decay",
                logger=LoggerConfig(project_name=PROJECT_NAME, entity=""),
                slice_keys=["num_kv_pairs", "input_seq_len", "test_type"],
                sweep_id=sweep_name,
                run_id=rid,
            ))

        # ===========================================
        # Mamba-2
        # ===========================================
        m2_mixer = dict(
            name="zoology.mixers.mamba2.Mamba2",
            kwargs={"d_state": d_state, "headdim": ssm_headdim},
        )
        m2_model = ModelConfig(
            block_type="Mamba2Block",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=m2_mixer,
            max_position_embeddings=0,
            name="mamba2",
            **model_factory_kwargs,
        )
        for lr in lr_sweep_mamba:
            rid = f"mamba2-h{nheads}-d{d_model}-lr{lr:.1e}"
            if rid in SKIP_RUNS:
                print(f"  ⏭  Skipping {rid} (already done)")
                continue
            configs.append(TrainConfig(
                model=m2_model, data=data, learning_rate=lr,
                max_epochs=8, lr_scheduler_type="linear_decay",
                logger=LoggerConfig(project_name=PROJECT_NAME, entity=""),
                slice_keys=["num_kv_pairs", "input_seq_len", "test_type"],
                sweep_id=sweep_name,
                run_id=rid,
            ))

        # ===========================================
        # RWKV-7
        # ===========================================
        rwkv_mixer = dict(
            name="zoology.mixers.rwkv7.RWKV7Attention",
            kwargs={
                "head_dim": rwkv_headdim,
                "num_hidden_layers": 2,
            },
        )
        rwkv_model = ModelConfig(
            block_type="RWKV7Block",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=rwkv_mixer,
            max_position_embeddings=0,
            name="rwkv7",
            **model_factory_kwargs,
        )
        for lr in LR_SWEEP_RWKV:
            rid = f"rwkv7-h{nheads}-d{d_model}-lr{lr:.1e}"
            if rid in SKIP_RUNS:
                print(f"  ⏭  Skipping {rid} (already done)")
                continue
            configs.append(TrainConfig(
                model=rwkv_model, data=data, learning_rate=lr,
                max_epochs=8, lr_scheduler_type="linear_decay",
                logger=LoggerConfig(project_name=PROJECT_NAME, entity=""),
                slice_keys=["num_kv_pairs", "input_seq_len", "test_type"],
                sweep_id=sweep_name,
                run_id=rid,
            ))

        # ===========================================
        # RWKV-7 PoST
        # ===========================================
        post_rwkv_mixer = dict(
            name="zoology.mixers.post_rwkv7_mixer.RWKV7PoSTMixer",
            kwargs={
                "head_dim": rwkv_headdim,
                "train_length": TRAIN_SEQ_LEN,
            },
        )
        post_rwkv_model = ModelConfig(
            block_type="RWKV7PoSTBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=post_rwkv_mixer,
            max_position_embeddings=0,
            name="rwkv7_post",
            **model_factory_kwargs,
        )
        for lr in LR_SWEEP_RWKV:
            rid = f"rwkv7_post-h{nheads}-d{d_model}-lr{lr:.1e}"
            if rid in SKIP_RUNS:
                print(f"  ⏭  Skipping {rid} (already done)")
                continue
            configs.append(TrainConfig(
                model=post_rwkv_model, data=data, learning_rate=lr,
                max_epochs=8, lr_scheduler_type="linear_decay",
                logger=LoggerConfig(project_name=PROJECT_NAME, entity=""),
                slice_keys=["num_kv_pairs", "input_seq_len", "test_type"],
                sweep_id=sweep_name,
                run_id=rid,
            ))

        # ===========================================
        # GDN (Gated DeltaNet)
        # ===========================================
        gdn_mixer = dict(
            name="zoology.mixers.gated_delta_net.GatedDeltaNet",
            kwargs={
                "num_heads": nheads,
                "expand_v": 1,
                "use_gate": False,
                "use_short_conv": True,
                "conv_size": 4,
            },
        )
        gdn_model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=gdn_mixer,
            max_position_embeddings=0,
            name="gdn",
            **model_factory_kwargs,
        )
        for lr in LR_SWEEP_GDN:
            rid = f"gdn-h{nheads}-d{d_model}-lr{lr:.1e}"
            if rid in SKIP_RUNS:
                print(f"  ⏭  Skipping {rid} (already done)")
                continue
            configs.append(TrainConfig(
                model=gdn_model, data=data, learning_rate=lr,
                max_epochs=8, lr_scheduler_type="linear_decay",
                logger=LoggerConfig(project_name=PROJECT_NAME, entity=""),
                slice_keys=["num_kv_pairs", "input_seq_len", "test_type"],
                sweep_id=sweep_name,
                run_id=rid,
            ))

        # ===========================================
        # GDN PoST (Gated DeltaNet PoST)
        # ===========================================
        gdn_post_mixer = dict(
            name="zoology.mixers.gated_delta_net_post.GatedDeltaNetPoST",
            kwargs={
                "num_heads": nheads,
                "expand_v": 1,
                "use_gate": False,
                "use_short_conv": True,
                "conv_size": 4,
                "post_mode": "adaptive",
                "train_length": TRAIN_SEQ_LEN,
                "position_adaptive": True,
                "alpha_mode": "analytical",
            },
        )
        gdn_post_model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=gdn_post_mixer,
            max_position_embeddings=0,
            name="gdn_post",
            **model_factory_kwargs,
        )
        for lr in LR_SWEEP_GDN:
            rid = f"gdn_post-h{nheads}-d{d_model}-lr{lr:.1e}"
            if rid in SKIP_RUNS:
                print(f"  ⏭  Skipping {rid} (already done)")
                continue
            configs.append(TrainConfig(
                model=gdn_post_model, data=data, learning_rate=lr,
                max_epochs=8, lr_scheduler_type="linear_decay",
                logger=LoggerConfig(project_name=PROJECT_NAME, entity=""),
                slice_keys=["num_kv_pairs", "input_seq_len", "test_type"],
                sweep_id=sweep_name,
                run_id=rid,
            ))

        # ===========================================
        # GLA (Gated Linear Attention)
        # ===========================================
        gla_mixer = dict(
            name="zoology.mixers.gla.GatedLinearAttention",
            kwargs={
                "num_heads": nheads,
                "expand_k": 1.0,
                "expand_v": 1.0,
                "use_short_conv": True,
                "conv_size": 4,
                "use_output_gate": True,
            },
        )
        gla_model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=gla_mixer,
            max_position_embeddings=0,
            name="gla",
            **model_factory_kwargs,
        )
        for lr in LR_SWEEP_GLA_RETNET:
            rid = f"gla-h{nheads}-d{d_model}-lr{lr:.1e}"
            if rid in SKIP_RUNS:
                print(f"  ⏭  Skipping {rid} (already done)")
                continue
            configs.append(TrainConfig(
                model=gla_model, data=data, learning_rate=lr,
                max_epochs=8, lr_scheduler_type="linear_decay",
                logger=LoggerConfig(project_name=PROJECT_NAME, entity=""),
                slice_keys=["num_kv_pairs", "input_seq_len", "test_type"],
                sweep_id=sweep_name,
                run_id=rid,
            ))

        # ===========================================
        # RetNet (Multi-Scale Retention)
        # ===========================================
        retnet_mixer = dict(
            name="zoology.mixers.retnet.RetNet",
            kwargs={
                "num_heads": nheads,
                "expand_k": 1.0,
                "expand_v": 1.0,
                "use_short_conv": True,
                "conv_size": 4,
                "use_output_gate": True,
            },
        )
        retnet_model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=retnet_mixer,
            max_position_embeddings=0,
            name="retnet",
            **model_factory_kwargs,
        )
        for lr in LR_SWEEP_GLA_RETNET:
            rid = f"retnet-h{nheads}-d{d_model}-lr{lr:.1e}"
            if rid in SKIP_RUNS:
                print(f"  ⏭  Skipping {rid} (already done)")
                continue
            configs.append(TrainConfig(
                model=retnet_model, data=data, learning_rate=lr,
                max_epochs=8, lr_scheduler_type="linear_decay",
                logger=LoggerConfig(project_name=PROJECT_NAME, entity=""),
                slice_keys=["num_kv_pairs", "input_seq_len", "test_type"],
                sweep_id=sweep_name,
                run_id=rid,
            ))

        # ===========================================
        # GLA PoST (= RetNet PoST under PoST parameterization)
        # Only GLA PoST is run; RetNet PoST collapses to the same model.
        # ===========================================
        gla_post_mixer = dict(
            name="zoology.mixers.gla_post.GLAPoST",
            kwargs={
                "num_heads": nheads,
                "expand_k": 1.0,
                "expand_v": 1.0,
                "use_short_conv": True,
                "conv_size": 4,
                "use_output_gate": True,
                "train_length": TRAIN_SEQ_LEN,
                "position_adaptive": True,
                "alpha_mode": "analytical",
            },
        )
        gla_post_model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=gla_post_mixer,
            max_position_embeddings=0,
            name="gla_post",
            **model_factory_kwargs,
        )
        for lr in LR_SWEEP_GLA_RETNET:
            rid = f"gla_post-h{nheads}-d{d_model}-lr{lr:.1e}"
            if rid in SKIP_RUNS:
                print(f"  ⏭  Skipping {rid} (already done)")
                continue
            configs.append(TrainConfig(
                model=gla_post_model, data=data, learning_rate=lr,
                max_epochs=8, lr_scheduler_type="linear_decay",
                logger=LoggerConfig(project_name=PROJECT_NAME, entity=""),
                slice_keys=["num_kv_pairs", "input_seq_len", "test_type"],
                sweep_id=sweep_name,
                run_id=rid,
            ))


# ============================================================
# Summary
# ============================================================
print(f"📊 Total configs: {len(configs)} (after skipping {len(SKIP_RUNS)} completed)")
print(f"   Models: Mamba-2 PoST, Mamba-2, RWKV-7 PoST, RWKV-7, GDN PoST, GDN, GLA PoST, GLA, RetNet")
print(f"   State sizes:")
for d in D_MODELS:
    for nh in NHEADS_BY_DMODEL[d]:
        ds = d // (2 * nh)
        state = d * d // nh
        print(f"     d={d}, nheads={nh}  →  state={state//1024}K  (d_state={ds})")
print(f"   Train: T={TRAIN_SEQ_LEN}, kv curriculum: {kv_schedule}")
print(f"   Test:  T={TEST_SEQ_LENS}, fixed kv={FIXED_KV}")
print(f"   Skipped: {len(SKIP_RUNS)} already-done runs")
