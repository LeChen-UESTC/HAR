# Skeleton Embedding HAR

- 项目根目录: `/data/chenle/GIRCSE/HAR`
- Conda 环境: `/data/chenle/.conda/envs/gircse`
- Embedding 模型: `/data/chenle/GIRCSE/HAR/models/Qwen3Embedding4B`
- 结构化描述: `/data/chenle/GIRCSE/HAR/data/cache/descriptions.json`
- Shift-GCN checkpoint: `/data/chenle/GIRCSE/HAR/models`
- NTU60 npz: `/data/chenle/GIRCSE/HAR/data/ntu_60/NTU_60.npz`
- NTU120 npz: `/data/chenle/GIRCSE/HAR/data/ntu_120/NTU120.npz`

当前代码只保留 embedding-only 流程。
所有文本原型和 skeleton-prefix reader 都基于 `Qwen3Embedding4B`。

## 配置

主配置：

```bash
configs/har_experiment.yaml
```

常用 projector 变体：

```bash
configs/har_experiment_structured_general.yaml
configs/har_experiment_structured_part_aware.yaml
configs/har_experiment_structured_linear.yaml
```

最常改字段：

```yaml
experiment.active_split: NTU55_5  # NTU55_5, NTU48_12, NTU110_10, NTU96_24
runtime.cuda_visible_devices: "0"
paths.embedding_model: /data/chenle/GIRCSE/HAR/models/Qwen3Embedding4B
paths.description_cache: "{project_root}/data/cache/descriptions.json"
train.stage: skeleton_embedding  # prealign, skeleton_embedding, direct_qformer_baseline
eval.task: zsl  # zsl, gzsl
```

## 缓存 Text Bank

先确认 `paths.description_cache` 已存在并包含结构化字段：

```text
label
observable_motion
key_body_parts
temporal_phases.start / middle / end
```

`descriptions.json` 支持 list of records，也支持 `{label: record}` 字典形式。

然后缓存 text bank：

```bash
python scripts/cache_text_bank.py --config configs/har_experiment.yaml
```

缓存会生成 `Zlabel`、`Zmotion`、`Zphase`、`Zmain`。`Zmain` 默认是
`Norm(0.7 * Zlabel + 0.3 * Zmotion)`。

## 训练

第一阶段 prealign：

```bash
python scripts/train.py --config configs/har_experiment_structured_part_aware.yaml \
  --override train.stage=prealign
```

第二阶段 skeleton embedding：

```bash
python scripts/train.py --config configs/har_experiment_structured_part_aware.yaml \
  --override train.stage=skeleton_embedding \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/train_prealign_NTU_55_5_BS128_EP10_structured_part_aware_qformer/last.ckpt
```

Direct baseline：

```bash
python scripts/train.py --config configs/har_experiment_structured_part_aware.yaml \
  --override train.stage=direct_qformer_baseline
```

## 评估

ZSL：

```bash
python scripts/eval.py --config configs/har_experiment_structured_part_aware.yaml \
  --override eval.task=zsl \
  --checkpoint /path/to/stage2/last.ckpt
```

GZSL：

```bash
python scripts/eval.py --config configs/har_experiment_structured_part_aware.yaml \
  --override eval.task=gzsl \
  --checkpoint /path/to/stage2/last.ckpt
```

## 输出

训练输出目录会带 `text_mode` 和 `projector_mode` 后缀，例如：

```text
outputs/models/train_skeleton_embedding_NTU_55_5_BS1_EP20_structured_part_aware_qformer/
```

checkpoint 只保存可训练的 skeleton-side 参数：

```text
shift_gcn.*
token_projector.*
embedding_projection.*
embedding_head.*  # direct_qformer_baseline only
```

冻结的 `Qwen3Embedding4B` 参数不会写入 checkpoint。
