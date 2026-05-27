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

