# 第二阶段：Skeleton-GIRCSE 训练

命令：

```bash
python scripts/train.py --config configs/har_experiment.yaml \
  --override train.stage=skeleton_gircse \
  --checkpoint /path/to/warmup/last.ckpt
```

常改字段：

```yaml
train.stage: skeleton_gircse
model.projector.type
model.soft_tokens.k_train
model.soft_tokens.pooling
train_presets.skeleton_gircse.train.epochs
train_presets.skeleton_gircse.train.batch_size
train_presets.skeleton_gircse.train.gradient_accumulation_steps
train_presets.skeleton_gircse.train.gradient_checkpointing
train_presets.skeleton_gircse.train.lr_projector
train_presets.skeleton_gircse.train.lr_shift_gcn
train_presets.skeleton_gircse.train.freeze_shift_gcn
train_presets.skeleton_gircse.train.freeze_llm
train_presets.skeleton_gircse.loss.temperature
train_presets.skeleton_gircse.loss.lambda_irr
```

`k_train` 支持单值：

```yaml
model.soft_tokens.k_train: 5
```

也支持 sweep：

```yaml
model.soft_tokens.k_train: [1, 5, 10, 20]
```

说明：

- sweep 只通过 `scripts/train.py` 生效。
- 每个 K 会独立训练一次，输出目录区分为 `K1`、`K5`、`K10`、`K20`。
- 如果手动传 `--exp_name xxx`，sweep 会自动改成 `xxx_K1`、`xxx_K5` 等，避免覆盖。
- 不要直接用 `scripts/train_skeleton_gircse.py` 跑 `k_train` 数组。
- `model.soft_tokens.k_test` 只用于评估。

`model.projector.type` 可选：

```text
linear
linear_layernorm
qformer
general_qformer
part_aware_qformer
```

`part_aware_qformer` 约束：

- `num_query_tokens == len(query_roles)`
- `len(joint_part_roles) == num_joints`
