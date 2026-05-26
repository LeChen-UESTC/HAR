# Skeleton-GIRCSE

This repository trains and evaluates Skeleton-GIRCSE for zero-shot skeleton action
recognition on NTU60/NTU120 SynSE/PURLS splits.

Default server paths:

- HAR project root: `/data/chenle/GIRCSE/HAR`
- Conda environment: `/data/chenle/.conda/envs/gircse`
- Qwen base model: `/data/chenle/GIRCSE/Qwen2.5-7B`
- GIRCSE LoRA adapter: `/data/chenle/GIRCSE/GIRCSE-QWEN7B`
- Shift-GCN checkpoints: `/data/chenle/GIRCSE/HAR/models`
- NTU60 npz: `/data/chenle/GIRCSE/HAR/data/ntu_60/NTU_60.npz`
- NTU120 npz: `/data/chenle/GIRCSE/HAR/data/ntu_120/NTU120.npz`

`GIRCSE-QWEN7B` is a LoRA adapter, not a standalone base model. The code loads
the local Qwen base model first, then attaches the GIRCSE adapter.

## Current Project Review

Problems found before the cleanup:

- Configs were split by dataset, split, stage, eval task, and projector ablation.
  Running one experiment required choosing from many similar YAML files.
- Several `base_config` values used absolute `/data/chenle/...` paths, making the
  configs hard to move or review locally.
- CUDA selection was outside the config and depended on shell prefixes such as
  `CUDA_VISIBLE_DEVICES=1,2`.
- Training/eval parameter names were inconsistent. For example, train used
  `batch_size`, eval used another nested `batch_size`, and eval-during-train was
  controlled by script flags or `eval_during_train`.
- Stage 1 saved only `last.ckpt`; it did not keep every epoch checkpoint.
- Run metadata recorded the latest checkpoint registry, but not a per-run
  `started_at`/`ended_at` lifecycle file.
- Evaluation required choosing one of several config files and one of several
  scripts, which made ZSL/GZSL/K-scaling easy to mix up.

The current recommended path is now one config plus one train command and one eval
command.

## Single Config

Use:

```bash
configs/har_experiment.yaml
```

Edit only these fields for common runs:

```yaml
project:
  root: /data/chenle/GIRCSE/HAR

experiment:
  active_split: NTU55_5  # NTU55_5, NTU48_12, NTU110_10, NTU96_24

runtime:
  cuda_visible_devices: "1,2"  # or null to use the current shell environment
  device: cuda

train:
  stage: skeleton_gircse  # prealign or skeleton_gircse

eval:
  task: zsl  # zsl, gzsl, or k_scaling
```

Training parameters live under `train_presets.<stage>.train` in the same file.
The important exposed knobs are:

- `batch_size`
- `gradient_accumulation_steps`
- `epochs`
- `eval_on_train`
- `eval_steps`
- `eval_every_epochs`
- `save_freq`
- `mixed_precision`
- `lr_projector`
- `lr_shift_gcn`

Evaluation parameters live under `eval_presets.<task>.eval`:

- `eval_batch_size`
- `k`
- `k_values`
- `sample_scope`
- `candidate_scope`
- `calibrated_stacking_gamma`

The four supported splits and their class lists are embedded in
`dataset_splits`, so changing `experiment.active_split` is enough to switch
between NTU55_5, NTU48_12, NTU110_10, and NTU96_24.

Historical YAML files under `configs/legacy/` are kept only for reproducing
previous runs. New runs should use `configs/har_experiment.yaml`.

## Train

Stage 1 warmup:

```bash
python scripts/train.py --config configs/har_experiment.yaml \
  --override train.stage=prealign
```

Stage 2 Skeleton-GIRCSE:

```bash
python scripts/train.py --config configs/har_experiment.yaml \
  --override train.stage=skeleton_gircse \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/train_NTU_55_5_BS16_EP10_K5/last.ckpt
```

You can also edit `train.stage` directly in `configs/har_experiment.yaml` and
omit the override.

## Evaluate

ZSL:

```bash
python scripts/eval.py --config configs/har_experiment.yaml \
  --override eval.task=zsl \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/train_NTU_55_5_BS1_EP20_K5/epoch_8.ckpt
```

GZSL:

```bash
python scripts/eval.py --config configs/har_experiment.yaml \
  --override eval.task=gzsl \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/train_NTU_55_5_BS1_EP20_K5/last.ckpt
```

K scaling:

```bash
python scripts/eval.py --config configs/har_experiment.yaml \
  --override eval.task=k_scaling \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/train_NTU_55_5_BS1_EP20_K5/last.ckpt
```

## Outputs

Training output directories use readable names:

```text
outputs/models/train_NTU_55_5_BS1_EP20_K5/
```

Evaluation output directories use the eval task:

```text
outputs/eval/eval_zsl_NTU_55_5_BS16_K10/
```

Each training run writes:

```text
epoch_1.ckpt
epoch_2.ckpt
...
last.ckpt
metrics.jsonl
config.yaml
run_meta.json
```

Checkpoints save only trainable skeleton-side parameters under `shift_gcn.*` and
`token_projector.*`. Frozen Qwen/GIRCSE weights are not written into epoch
checkpoints. If `train.freeze_shift_gcn=true`, the frozen Shift-GCN parameters
are also excluded.

`run_meta.json` records:

- `started_at`
- `ended_at`
- `duration_seconds`
- `status`
- `active_split`
- `train_stage`
- `eval_task`
- `cuda_visible_devices`
- `command`

`outputs/models/all/latest_run.json` and `outputs/models/all/runs.jsonl` still
track the latest Stage 2 checkpoint and per-epoch registry entries.

## Text Bank

Rich descriptions and text embeddings are shared by NTU60/NTU120 splits:

```bash
python scripts/generate_rich_description.py --config configs/har_experiment.yaml
python scripts/cache_text_bank.py --config configs/har_experiment.yaml
```

NTU60 reuses the NTU120 text bank and selects only the classes needed by the
active split.
