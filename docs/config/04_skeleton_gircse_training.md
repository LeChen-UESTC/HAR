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
train_presets.skeleton_gircse.train.text_bank_path
train_presets.skeleton_gircse.loss.temperature
train_presets.skeleton_gircse.loss.lambda_motion
train_presets.skeleton_gircse.loss.lambda_phase
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
- 每个 K 会独立训练一次，输出目录区分为 `K1`、`K5`、`K10`、`K20`，并带 `text_mode` 和 `projector_mode` 后缀。
- 如果手动传 `--exp_name xxx`，sweep 会自动改成 `xxx_K1_structured_part_aware_qformer`、`xxx_K5_structured_part_aware_qformer` 等，避免覆盖。
- 不要直接用 `scripts/train_skeleton_gircse.py` 跑 `k_train` 数组。
- `model.soft_tokens.k_test` 只用于评估。
- `text_bank_path: null` 表示使用全局 `paths.text_bank`；填路径则只覆盖第二阶段。
- 加载 text bank 时会校验其 metadata 中的 `text_mode` 是否等于当前配置。
- 加载 checkpoint 时会校验 `text_mode` 和 `projector_type` 是否等于当前配置。
- Skeleton-GIRCSE 的 LLM 输入顺序是 `[T_skel; prompt]`。
- 当前 prompt 是 `Instruct: Represent the semantic meaning of the preceding human skeleton motion for zero-shot action recognition.\nRepresentation:`。
- 当前 Q-Former 输出 `16` 个 token：6 part、3 phase、1 global、6 free。
- 当前默认 loss：`Lmain + 0.3 Lmotion + 0.2 Lphase + 0.1 LIRR`。

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

Baseline 阶段：

```text
direct_qformer_baseline: T_skel -> MeanPool -> MLP -> z_direct
anchor_hidden_baseline: [T_skel; prompt] -> prompt 最后 token hidden state
```
