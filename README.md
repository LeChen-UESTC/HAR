# Skeleton-GIRCSE

本项目是 Skeleton-GIRCSE 的本地工程骨架，用于在服务器
`/data/chenle/GIRCSE` 下复现实验。所有会影响结果的路径、模型选择、采样策略、
loss、projection、训练阶段与评估开关都通过配置或命令行传入。

## GIRCSE 官方代码来源

GIRCSE 相关 soft-token 生成逻辑参考并 vendor 了官方实现：

- Source: https://github.com/Roytsai27/GIRCSE/tree/main/embedding
- Commit: `20676c15294e161bcfd5d5be97e75498e54fdb8f`
- Vendored path: `src/third_party/gircse_embedding/`
- License: MIT, copied in `src/third_party/gircse_embedding/LICENSE`

项目实际调用的适配层是 `src/models/gircse_adapter.py`，它保留官方
`BaseReasoningTrainer.encode()` / `_extend_sequence()` / `GIRCSETrainer.get_next_token_embedding()`
的核心机制，同时支持 skeleton projected tokens 以 `inputs_embeds` 形式进入 LLM。

## BLIP-2 Q-Former 来源

Q-Former projector 使用 Salesforce LAVIS 的 BLIP-2 Q-Former 实现：

- Source: https://github.com/salesforce/LAVIS/tree/main/lavis/models/blip2_models
- Commit: `506965b9c4a18c1e565bd32acaccabe0198433f7`
- Vendored path: `src/third_party/lavis_blip2_qformer/`
- License: BSD-3-Clause, copied in `src/third_party/lavis_blip2_qformer/LICENSE.txt`

默认 projector 是 `part_aware_qformer`，使用 7 个 query：
`head`、`left_arm`、`right_arm`、`torso`、`left_leg`、`right_leg`、`global`。
本版本只做 Part-aware Query Initialization + Global Cross-Attention，不启用 `L_part` 弱监督。

## Shift-GCN 官方代码来源

骨架 encoder 默认使用官方 Shift-GCN 结构：

- Source: https://github.com/kchengiva/Shift-GCN
- Vendored path: `src/third_party/shift_gcn/`
- License: Creative Commons Attribution-NonCommercial 4.0 International,
  copied in `src/third_party/shift_gcn/LICENSE.txt`

本仓库保留官方 `l1`-`l10`、Shift-GCN spatial shift、分类头命名，以兼容官方
`.pt` 权重；同时新增 `forward_features()`，输出进入 Skeleton Q-Former 的特征图。
官方仓库的旧 CUDA temporal shift 扩展被替换为纯 PyTorch fallback，避免服务器重新编译
PyTorch 0.4/CUDA 9 时代的插件。

## 服务器模型路径

默认配置已写入服务器路径：

- GIRCSE-Qwen7B: `/data/chenle/GIRCSE/GIRCSE-QWEN7B`
- Qwen2.5-7B-Instruct: `/data/chenle/GIRCSE/Qwen2.5-7B`
- NTU120 official Shift-GCN xsub checkpoint:
  `/data/chenle/GIRCSE/HAR/models/shift_gcn_ntu120_xsub.pt`
- NTU60 official Shift-GCN xsub checkpoint:
  `/data/chenle/GIRCSE/HAR/models/shift_gcn_ntu60_xsub.pt`

`GIRCSE-Qwen7B` 是 LoRA adapter 目录，不是完整基座模型。代码会显式加载
`/data/chenle/GIRCSE/Qwen2.5-7B` 作为本地 base model，再挂载
`/data/chenle/GIRCSE/GIRCSE-QWEN7B` adapter，避免服务器无外网时误连 Hugging Face Hub。
Qwen2.5-7B 的 hidden size 是 3584，因此 projector `llm_dim` 和 text bank embedding
维度都配置为 3584。

本地不要求存在这些模型目录；部署到服务器后按配置运行即可。

默认把官方 Shift-GCN 发布权重作为冻结 skeleton encoder 使用。权重文件不提交到 Git，
服务器上需要先放到上述 `models/` 路径。如果从官方仓库下载，可按下面的命名拷贝：

```bash
mkdir -p /data/chenle/GIRCSE/HAR/models
cp /path/to/Shift-GCN/save_models/ntu120_ShiftGCN_joint_xsub.pt \
  /data/chenle/GIRCSE/HAR/models/shift_gcn_ntu120_xsub.pt
cp /path/to/Shift-GCN/save_models/ntu_ShiftGCN_joint_xsub.pt \
  /data/chenle/GIRCSE/HAR/models/shift_gcn_ntu60_xsub.pt
```

默认数据配置使用已经预处理好的 NTU `.npz` 文件：

- NTU120: `/data/chenle/GIRCSE/HAR/data/ntu_120/NTU120.npz`
- NTU60: `/data/chenle/GIRCSE/HAR/data/ntu_60/NTU_60.npz`

`.npz` 内部应包含 `x_data` 与 `y_data`，其中 `x_data` 采用 `[N, T, M*V*C]`
布局，训练时会转换为 Shift-GCN 的 `[C, T, V, M]`。

## 环境

用户指定的 Python 环境：

```bash
source /Users/bytedance/.pyenv/versions/3.10.15/envs/env310/bin/activate
```

本仓库不会自动安装或修改依赖。服务器部署时可按需执行：

```bash
pip install -r requirements.txt
```

## 快速启动

已提供四个 PURLS/SynSE-style ZSL 类别划分配置：

```text
configs/ntu60_zsl_55_5.yaml
configs/ntu60_zsl_48_12.yaml
configs/ntu120_zsl_110_10.yaml
configs/ntu120_zsl_96_24.yaml
```

`configs/train_warmup.yaml`、`configs/train_gircse.yaml` 和 projector 消融配置默认继承
`configs/ntu120_zsl_110_10.yaml`，只作为默认别名保留。正式运行建议使用下面这些显式
Stage 1/Stage 2 配置，避免 NTU60/NTU120 和 split 混用：

```text
configs/train_warmup_ntu120_110_10.yaml
configs/train_gircse_ntu120_110_10.yaml
configs/train_warmup_ntu120_96_24.yaml
configs/train_gircse_ntu120_96_24.yaml
configs/train_warmup_ntu60_55_5.yaml
configs/train_gircse_ntu60_55_5.yaml
configs/train_warmup_ntu60_48_12.yaml
configs/train_gircse_ntu60_48_12.yaml
```

生成全集富文本描述。该步骤只依赖 `index_action_map.json` 的 120 个动作标签，
不依赖具体 ZSL split；生成结果供 NTU60/NTU120 及所有 split 共享。

```bash
python scripts/generate_rich_description.py --config configs/ntu120_zsl.yaml
```

缓存文本 embedding：

```bash
python scripts/cache_text_bank.py --config configs/ntu120_zsl_110_10.yaml
```

NTU60 直接复用上述 NTU120 text bank；配置会按 split 中的 class id 只取前 60 类里的
seen/unseen 候选，不需要额外生成 `text_embeddings_ntu60_zsl.pt`。

Stage 1/Stage 2 默认冻结 Shift-GCN，只训练 Skeleton Q-Former projector。

Stage 1 预对齐 warmup：

```bash
python scripts/train_prealign.py --config configs/train_warmup_ntu120_110_10.yaml
```

Stage 2 Skeleton-GIRCSE 训练：

```bash
CUDA_VISIBLE_DEVICES=1,2 python scripts/train_skeleton_gircse.py \
  --config configs/train_gircse_ntu60_55_5.yaml \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/prealign-ntu60-split_55_5-modality_skeleton-loss_classwise_infonce-proj_part_aware_qformer-dim_3584-K_5-6e50a0d2fe-20260519_083938/last.ckpt \
  --wandb_mode offline
```

Stage 2 会反向穿过冻结的 Qwen/GIRCSE 到 skeleton prefix，因此默认使用
`device_map_train=auto`、`batch_size=1`、`gradient_accumulation_steps=8` 和
gradient checkpointing 来控制显存；等效 batch size 仍是 8。建议至少暴露两张空闲 GPU：

```bash
CUDA_VISIBLE_DEVICES=1,2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_skeleton_gircse.py \
  --config configs/train_gircse_ntu60_55_5.yaml \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/prealign-ntu60-split_55_5-modality_skeleton-loss_classwise_infonce-proj_part_aware_qformer-dim_3584-K_5-6e50a0d2fe-20260519_083938/last.ckpt \
  --wandb_mode offline
```

Projector 消融配置：

```bash
python scripts/train_skeleton_gircse.py --config configs/projector_linear.yaml --wandb_mode offline
python scripts/train_skeleton_gircse.py --config configs/projector_general_qformer.yaml --wandb_mode offline
python scripts/train_skeleton_gircse.py --config configs/projector_part_aware_qformer.yaml --wandb_mode offline
```

ZSL/GZSL 评估：

`eval_zsl.py` 和 `eval_gzsl.py` 使用当前模型配置中的单个 `K` 评估；如果要测试
`K=1,3,5`，使用 `eval_k_scaling.py`。

```bash
CUDA_VISIBLE_DEVICES=1,2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/eval_zsl.py \
  --config configs/train_gircse_ntu60_55_5.yaml \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/skeleton_gircse-ntu60-split_55_5-modality_skeleton-loss_stepwise_infonce_irr-proj_part_aware_qformer-dim_3584-K_5-eade367866-20260519_141225/last.ckpt \
  --wandb_mode offline

CUDA_VISIBLE_DEVICES=1,2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/eval_gzsl.py \
  --config configs/train_gircse_ntu60_55_5.yaml \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/skeleton_gircse-ntu60-split_55_5-modality_skeleton-loss_stepwise_infonce_irr-proj_part_aware_qformer-dim_3584-K_5-eade367866-20260519_141225/last.ckpt \
  --wandb_mode offline

CUDA_VISIBLE_DEVICES=1,2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/eval_k_scaling.py \
  --config configs/train_gircse_ntu60_55_5.yaml \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/skeleton_gircse-ntu60-split_55_5-modality_skeleton-loss_stepwise_infonce_irr-proj_part_aware_qformer-dim_3584-K_5-eade367866-20260519_141225/last.ckpt \
  --override model.soft_tokens.k_test=1,3,5 \
  --wandb_mode offline
```

绘制本地曲线：

```bash
python visualization/plot_curves.py --log logs/experiment_latest.log
```

GIRCSE soft token 可视化导出：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/export_gircse_soft_tokens.py \
  --base_model_path /data/chenle/GIRCSE/Qwen2.5-7B \
  --adapter_path /data/chenle/GIRCSE/GIRCSE-QWEN7B \
  --include_base \
  --text "Why is it so hard to track down this card?" \
  --instruction "Represent the intention of this text." \
  --instruction_name intention \
  --instruction "Represent the emotion of this text." \
  --instruction_name emotion \
  --k 20 \
  --topk 30 \
  --raw_topk 500 \
  --output_json visualization/logs/gircse_soft_tokens_table4.json
```

打开可视化页面：

```bash
python -m http.server 8000
```

然后访问 `http://localhost:8000/visualization/soft_token_viewer.html`，上传导出的 JSON。
页面支持查看每个 step 与 step group `1-5 / 6-10 / 11-20` 的 raw top tokens、
filtered semantic tokens、过滤 anchor 后重归一化的 residual semantic tokens。
可在页面中按 frequency、probability、rank 或 first step 切换排序。
示例 JSON 位于
`visualization/examples/gircse_soft_tokens_mock.json`。

## 输出规范

每次运行会创建实验目录：

```text
outputs/models/<exp_name>/
outputs/eval/<exp_name>/
logs/experiment_<timestamp>.log
```

其中 `<exp_name>` 由关键配置生成，包含数据集、模态、loss、projection、维度、K 等指纹。
完整配置会保存为 `config.yaml` 并同步到 WandB；WandB 不可用时自动降级到本地 offline/disabled。

## 缓存规范

缓存命名：

```text
{dataset_name}_{sampling_strategy_hash}_{preprocess_version}.lmdb
```

采样策略会写入 `cache_metadata.json`。启动时若发现缓存 metadata 与当前配置不匹配，
旧缓存会改名为 `.deprecated_<timestamp>`，训练自动回退到 raw 数据或触发重建逻辑。

缓存缺失时会输出：

```text
WARNING: cache missing for key X, falling back to raw data
```

坏样本或解码异常会记录到 `skipped_samples.log`，不会中断训练。

## 目录

```text
configs/            # 显式实验配置
scripts/            # Python 入口
scripts_sh/         # Shell 启动示例
src/                # 核心源码
data/ntu_60/        # 预处理后的 NTU60 npz
data/ntu_120/       # 预处理后的 NTU120 npz
data/cache/         # 预处理缓存
outputs/            # checkpoint、metrics、predictions
visualization/      # 本地曲线绘图
tests/              # 单元测试
logs/               # 结构化文本日志
```
