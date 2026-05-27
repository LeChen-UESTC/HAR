# HAR 配置说明

统一改 `configs/har_experiment.yaml`。

- `00_common.md`：所有阶段共用字段。
- `01_text_description_generation.md`：生成 rich description。
- `02_text_bank_embedding.md`：缓存 text bank。
- `03_warmup_prealign.md`：第一阶段 warmup / prealign。
- `04_skeleton_gircse_training.md`：第二阶段 Skeleton-GIRCSE 训练和 `k_train` sweep。
- `05_evaluation.md`：ZSL / GZSL / K-scaling 评估。

