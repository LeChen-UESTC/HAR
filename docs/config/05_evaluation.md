# 评估

命令：

```bash
python scripts/eval.py --config configs/har_experiment.yaml \
  --override eval.task=zsl \
  --checkpoint /path/to/stage2/last.ckpt

python scripts/eval.py --config configs/har_experiment.yaml \
  --override eval.task=gzsl \
  --checkpoint /path/to/stage2/last.ckpt
```

常改字段：

```yaml
eval.task
eval_presets.<task>.eval.sample_scope
eval_presets.<task>.eval.candidate_scope
eval_presets.<task>.eval.text_bank_path
eval_presets.<task>.eval.eval_batch_size
```

`eval.task` 可选：

```text
zsl
gzsl
```

`sample_scope` / `candidate_scope` 可选：

```text
seen
unseen
all
seen+unseen
gzsl
none
null
```

说明：

- `zsl` 默认只评估 unseen 类。
- `gzsl` 默认评估 seen + unseen，并报告 harmonic mean。
- 评估 prealign checkpoint 时必须加 `--override train.stage=prealign`；评估 skeleton embedding checkpoint 时使用 `--override train.stage=skeleton_embedding` 或默认配置。
- `text_bank_path: null` 表示使用全局 `paths.text_bank`；填路径则只覆盖当前评估任务。
- 评估输出目录和 `metrics.json` 会记录当前 `text_mode` 和 `projector_type`。
- 加载 text bank 时会校验其 metadata 中的 `text_mode` 是否等于当前配置。
- 加载 checkpoint 时会校验 `text_mode`、`projector_type`、`train_stage` 是否等于当前配置。
