# 缓存 Text Bank

命令：

```bash
python scripts/cache_text_bank.py --config configs/har_experiment.yaml
```

常改字段：

```yaml
paths.text_bank
paths.description_cache
paths.gircse_base_model
paths.gircse_adapter
text_branch.description_variant
text_branch.embedding.prompt
text_branch.embedding.k_text
text_branch.embedding.pooling
text_branch.embedding.logit_temperature
```

`description_variant` 可选：

```text
label_only
label_local_motion
label_local_motion_object
full
```

说明：

- `k_text` 是 text branch 的 GIRCSE soft-token 生成步数。
- `pooling` 可选 `generate_mean` 或 `last`。
- 改 description cache、GIRCSE 路径、`description_variant` 或 `text_branch.embedding.*` 后，应重新生成 `text_bank`。

