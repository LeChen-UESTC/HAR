# 第一阶段：Warmup / Prealign

命令：

```bash
python scripts/train.py --config configs/har_experiment.yaml --override train.stage=prealign
```

常改字段：

```yaml
train.stage: prealign
train_presets.prealign.train.epochs
train_presets.prealign.train.batch_size
train_presets.prealign.train.lr_projector
train_presets.prealign.train.lr_shift_gcn
train_presets.prealign.train.freeze_shift_gcn
train_presets.prealign.train.mixed_precision
train_presets.prealign.loss.temperature
```

训练内容：

- `freeze_shift_gcn: true` 时，只训练 `token_projector`。
- 第一阶段不加载 LLM，所以没有 `freeze_llm` / `freeze_lm_head`。
- `device_map_train: null` 表示整个 warmup model 直接放到 `runtime.device`。

如需 warmup 期间验证：

```yaml
train_presets.prealign.train.eval_on_train: true
train_presets.prealign.train.eval_every_epochs: 1
```

第二阶段用第一阶段输出的 `last.ckpt` 或 `best.ckpt` 作为 `--checkpoint`。

