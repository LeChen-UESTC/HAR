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
text_branch.embedding.main_label_alpha
text_branch.embedding.logit_temperature
```

`description_variant` 可选：

```text
structured
```

缓存会同时生成 4 个 bank：

```text
Zlabel: 只编码类别名文本
Zmotion: 编码 skeleton-observable motion 和关键身体部位
Zphase: 编码 start/middle/end 时序文本
Zmain: Norm(alpha * Zlabel + (1 - alpha) * Zmotion)
```

默认输出文件名会携带关键文本配置：

```yaml
paths.text_bank: "{project_root}/data/cache/text_embeddings_ntu{text_num_classes}_zsl{text_mode}_k{k_text}_{text_pooling}_a{main_label_alpha}.pt"
```

说明：

- `k_text` 是 text branch 的 GIRCSE soft-token 生成步数。
- `pooling` 可选 `generate_mean` 或 `last`。
- `main_label_alpha: 0.7` 表示 `Zmain` 中 70% 来自 `Zlabel`，30% 来自 `Zmotion`。
- `text_mode` 目前只有 `_structured`。
- 改 description cache、GIRCSE 路径、`description_variant` 或 `text_branch.embedding.*` 后，应重新生成 `text_bank`。
- 训练/评估加载 text bank 时会校验 `text_mode`、GIRCSE 路径、prompt、`k_text`、`pooling`、`main_label_alpha` 等 metadata。
