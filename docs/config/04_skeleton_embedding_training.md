# 第二阶段：Skeleton Embedding 训练

命令：

```bash
python scripts/train.py --config configs/har_experiment.yaml \
  --override train.stage=skeleton_embedding \
  --checkpoint /path/to/prealign/last.ckpt
```

常改字段：

```yaml
train.stage: skeleton_embedding
model.projector.type
train_presets.skeleton_embedding.train.epochs
train_presets.skeleton_embedding.train.batch_size
train_presets.skeleton_embedding.train.gradient_accumulation_steps
train_presets.skeleton_embedding.train.gradient_checkpointing
train_presets.skeleton_embedding.train.lr_projector
train_presets.skeleton_embedding.train.lr_shift_gcn
train_presets.skeleton_embedding.train.freeze_shift_gcn
train_presets.skeleton_embedding.train.freeze_embedding_model
train_presets.skeleton_embedding.train.text_bank_path
train_presets.skeleton_embedding.loss.temperature
train_presets.skeleton_embedding.loss.lambda_motion
train_presets.skeleton_embedding.loss.lambda_phase
```

`train.stage` 可选：

```text
prealign
skeleton_embedding
direct_qformer_baseline
```

说明：

- 主模型数据流：`Shift-GCN -> Projector -> T_skel -> [T_skel; prompt] -> Qwen3Embedding4B -> prompt 最后 token hidden state -> text-space projection`。
- 当前 prompt 是 `Instruct: Represent the semantic meaning of the preceding human skeleton motion for zero-shot action recognition.\nRepresentation:`。
- 当前 Q-Former 输出 `16` 个 token：6 part、3 phase、1 global、6 free。
- 默认 loss：`Lmain + 0.3 Lmotion + 0.2 Lphase`，其中 `Lmain` 使用 `Zmain`，辅助项使用 `Zmotion` 和 `Zphase`。
- `freeze_embedding_model: true` 表示 Qwen3Embedding4B 只前向读 skeleton prefix，不更新参数；训练的是 Shift-GCN（若未冻结）、Projector、text-space projection。
- `text_bank_path: null` 表示使用全局 `paths.text_bank`；填路径则只覆盖第二阶段。
- 加载 text bank 会校验 embedding 模型路径、prompt、pooling、text_mode 等 metadata。
- 加载 checkpoint 会校验 `text_mode` 和 `projector_type`。

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
```
