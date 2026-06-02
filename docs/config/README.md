# HAR 配置说明

共享参数在 `configs/har_experiment.yaml`。实际实验优先使用下面 3 个版本配置：

```text
configs/har_experiment_structured_general.yaml
configs/har_experiment_structured_part_aware.yaml
configs/har_experiment_structured_linear.yaml
```

这些版本只覆盖实验标签、GPU/批量大小（linear 版本）和 `model.projector.type`。文本描述统一使用 structured schema。

- `00_common.md`：所有阶段共用字段。
- `01_structured_text_descriptions.md`：结构化文本描述文件格式。
- `02_text_bank_embedding.md`：缓存 text bank。
- `03_warmup_prealign.md`：第一阶段 warmup / prealign。
- `04_skeleton_embedding_training.md`：第二阶段 Skeleton Embedding 训练和 baseline。
- `05_evaluation.md`：ZSL / GZSL 评估。
