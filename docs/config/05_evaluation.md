# 评估

命令：

```bash
python scripts/eval.py --config configs/har_experiment.yaml \
  --override eval.task=zsl \
  --checkpoint /path/to/stage2/last.ckpt

python scripts/eval.py --config configs/har_experiment.yaml \
  --override eval.task=gzsl \
  --checkpoint /path/to/stage2/last.ckpt

python scripts/eval.py --config configs/har_experiment.yaml \
  --override eval.task=k_scaling \
  --checkpoint /path/to/stage2/last.ckpt
```

常改字段：

```yaml
eval.task
eval_presets.zsl.eval.k
eval_presets.gzsl.eval.k
model.soft_tokens.k_test
eval_presets.<task>.eval.sample_scope
eval_presets.<task>.eval.candidate_scope
eval_presets.<task>.eval.text_bank_path
eval_presets.<task>.eval.eval_batch_size
```

`eval.task` 可选：

```text
zsl
gzsl
k_scaling
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
- `k_scaling` 使用 `model.soft_tokens.k_test` 对同一个 checkpoint 扫描多个 test-time K。
- `k_scaling` 只适用于 Skeleton-GIRCSE 生成式模型，不适用于 direct / anchor baseline。
- direct / anchor baseline 的 `eval.k` 会被忽略，因为它们不生成 soft token。
- `text_bank_path: null` 表示使用全局 `paths.text_bank`；填路径则只覆盖当前评估任务。
- 评估输出目录、`metrics.json` / `k_scaling_metrics.json` 都会记录当前 `text_mode` 和 `projector_type`。
- 加载 text bank 时会校验其 metadata 中的 `text_mode` 是否等于当前配置。
- 加载 checkpoint 时会校验 `text_mode` 和 `projector_type` 是否等于当前配置。
