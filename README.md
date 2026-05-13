# Skeleton-GIRCSE

本项目是 Skeleton-GIRCSE 的本地工程骨架，用于在服务器
`/home/chenle/GIRCSE` 下复现实验。所有会影响结果的路径、模型选择、采样策略、
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

## 服务器模型路径

默认配置已写入服务器路径：

- GIRCSE-Qwen7B: `/home/chenle/GIRCSE/GIRCSE-QWEN7B`
- Qwen2.5-7B-Instruct: `/home/chenle/GIRCSE/Qwen2.5-7B`

本地不要求存在这些模型目录；部署到服务器后按配置运行即可。

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

生成富文本描述：

```bash
python scripts/generate_rich_description.py --config configs/ntu60_zsl.yaml
```

缓存文本 embedding：

```bash
python scripts/cache_text_bank.py --config configs/ntu60_zsl.yaml
```

Stage 0 Shift-GCN 预训练：

```bash
python scripts/train_shiftgcn_seen.py --config configs/ntu60_zsl.yaml
```

Stage 1 预对齐 warmup：

```bash
python scripts/train_prealign.py --config configs/train_warmup.yaml
```

Stage 2 Skeleton-GIRCSE 训练：

```bash
python scripts/train_skeleton_gircse.py --config configs/train_gircse.yaml --wandb_mode offline
```

Projector 消融配置：

```bash
python scripts/train_skeleton_gircse.py --config configs/projector_linear.yaml --wandb_mode offline
python scripts/train_skeleton_gircse.py --config configs/projector_general_qformer.yaml --wandb_mode offline
python scripts/train_skeleton_gircse.py --config configs/projector_part_aware_qformer.yaml --wandb_mode offline
```

ZSL/GZSL 评估：

```bash
python scripts/eval_zsl.py --config configs/ntu60_zsl.yaml
python scripts/eval_gzsl.py --config configs/ntu60_zsl.yaml
python scripts/eval_k_scaling.py --config configs/ntu60_zsl.yaml
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
页面支持查看每个 step 的 raw top tokens、filtered semantic tokens，以及 step group
`1-5 / 6-10 / 11-20` 的去重语义词。示例 JSON 位于
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
data/src/           # 原始数据，建议软链接到外部存储
data/cache/         # 预处理缓存
outputs/            # checkpoint、metrics、predictions
visualization/      # 本地曲线绘图
tests/              # 单元测试
logs/               # 结构化文本日志
```
