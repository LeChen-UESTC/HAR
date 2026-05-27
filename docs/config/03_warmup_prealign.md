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
train_presets.prealign.train.text_bank_path
train_presets.prealign.train.mixed_precision
train_presets.prealign.loss.temperature
```

训练内容：

- `freeze_shift_gcn: true` 时，只训练 `token_projector`。
- 第一阶段不加载 LLM，所以没有 `freeze_llm` / `freeze_lm_head`。
- `device_map_train: null` 表示整个 warmup model 直接放到 `runtime.device`。
- `text_bank_path: null` 表示使用全局 `paths.text_bank`；填路径则只覆盖第一阶段。
- 输出目录会带 `text_mode` 后缀，例如 `_full`、`_label`。
- 加载 text bank 时会校验其 metadata 中的 `text_mode` 是否等于当前配置。

如需 warmup 期间验证：

```yaml
train_presets.prealign.train.eval_on_train: true
train_presets.prealign.train.eval_every_epochs: 1
```

第二阶段用第一阶段输出的 `last.ckpt` 或 `best.ckpt` 作为 `--checkpoint`。
