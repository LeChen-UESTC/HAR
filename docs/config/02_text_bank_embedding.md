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

默认输出文件名会携带关键文本配置：

```yaml
paths.text_bank: "{project_root}/data/cache/text_embeddings_ntu{text_num_classes}_zsl{text_mode}.pt"
```

例如：

```text
text_embeddings_ntu120_zsl_full.pt
text_embeddings_ntu120_zsl_label.pt
text_embeddings_ntu120_zsl_label_local_motion.pt
```

说明：

- `k_text` 是 text branch 的 GIRCSE soft-token 生成步数。
- `pooling` 可选 `generate_mean` 或 `last`。
- `text_mode` 是带下划线的产物后缀：`_full`、`_label`、`_label_local_motion`、`_label_local_motion_object`。
- 改 description cache、GIRCSE 路径、`description_variant` 或 `text_branch.embedding.*` 后，应重新生成 `text_bank`。
- 如果你要同时比较不同 `k_text` 或 `pooling`，手动把它们也加进 `paths.text_bank`。
- 保存的 text bank metadata 会记录 `description_variant` 和 `text_mode`，训练/评估加载时会做一致性校验。
