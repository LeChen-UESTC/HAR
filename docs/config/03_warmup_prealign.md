# 第一阶段：Warmup / Prealign

命令：

```bash
python scripts/train.py --config configs/har_experiment.yaml --override train.stage=prealign
```

两卡 DDP：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/har_experiment.yaml \
  --override train.stage=prealign
```

说明：`CUDA_VISIBLE_DEVICES=0,1` 只让两张卡可见；DDP 必须用 `torchrun` 启动多进程。

常改字段：

```yaml
train.stage: prealign
train_presets.prealign.train.epochs
train_presets.prealign.train.batch_size
train_presets.prealign.train.lr_projector
train_presets.prealign.train.lr_shift_gcn
train_presets.prealign.train.freeze_shift_gcn
train_presets.prealign.train.text_bank_path
train_presets.prealign.train.mixed_precision
train_presets.prealign.loss.temperature
train_presets.prealign.loss.lambda_motion
train_presets.prealign.loss.lambda_phase
```

训练内容：

- `freeze_shift_gcn: true` 时，只训练 projector；若 text bank 维度不同，还会训练 `embedding_projection`。
- 第一阶段不前向加载 Qwen3Embedding4B，只读取其 hidden size，所以没有 `freeze_embedding_model`。
- `device_map_train: null` 表示整个 warmup model 直接放到 `runtime.device`。
- `text_bank_path: null` 表示使用全局 `paths.text_bank`；填路径则只覆盖第一阶段。
- 输出目录会带 `text_mode` 和 `projector_mode` 后缀，例如 `_structured_part_aware_qformer`、`_structured_linear`。
- 加载 text bank 时会校验其 metadata 中的 `text_mode` 是否等于当前配置。
- 第一阶段 loss 使用 `Zmain`，并可按 `lambda_motion`、`lambda_phase` 加入 `Zmotion`、`Zphase` 辅助监督。
- 第二阶段加载 warmup checkpoint 时会校验 `text_mode` 和 `projector_type` 是否等于当前配置。

如需 warmup 期间验证：

```yaml
train_presets.prealign.train.eval_on_train: true
train_presets.prealign.train.eval_every_epochs: 1
```

第二阶段用第一阶段输出的 `last.ckpt` 或 `best.ckpt` 作为 `--checkpoint`。
