# Skeleton-GIRCSE
项目目录：/data/chenle/GIRCSE
chenle@ubuntu:~/GIRCSE/HAR$ tree -I outputs
.
├── configs
│   ├── eval_gzsl_ntu60_55_5.yaml
│   ├── eval_k_scaling_ntu60_55_5.yaml
│   ├── eval_zsl_ntu60_55_5.yaml
│   ├── ntu120_zsl_110_10.yaml
│   ├── ntu120_zsl_96_24.yaml
│   ├── ntu120_zsl.yaml
│   ├── ntu60_zsl_48_12.yaml
│   ├── ntu60_zsl_55_5.yaml
│   ├── ntu60_zsl.yaml
│   ├── projector_general_qformer.yaml
│   ├── projector_linear.yaml
│   ├── projector_part_aware_qformer.yaml
│   ├── train_gircse_ntu120_110_10.yaml
│   ├── train_gircse_ntu120_96_24.yaml
│   ├── train_gircse_ntu60_48_12.yaml
│   ├── train_gircse_ntu60_55_5.yaml
│   ├── train_gircse.yaml
│   ├── train_warmup_ntu120_110_10.yaml
│   ├── train_warmup_ntu120_96_24.yaml
│   ├── train_warmup_ntu60_48_12.yaml
│   ├── train_warmup_ntu60_55_5_general.yaml
│   ├── train_warmup_ntu60_55_5.yaml
│   └── train_warmup.yaml
├── data
│   ├── cache
│   │   ├── rich_descriptions_ntu120.json
│   │   └── text_embeddings_ntu120_zsl.pt
│   ├── index_action_map.json
│   ├── ntu_120
│   │   └── NTU120.npz
│   ├── ntu_60
│   │   └── NTU_60.npz
│   └── read_NTU60.py
├── models
│   ├── shift_gcn_ntu120_xsub.pt
│   └── shift_gcn_ntu60_xsub.pt
├── README.md
├── scripts
│   ├── _bootstrap.py
│   ├── cache_text_bank.py
│   ├── eval_gzsl.py
│   ├── eval_k_scaling.py
│   ├── eval_zsl.py
│   ├── export_gircse_soft_tokens.py
│   ├── generate_rich_description.py
│   ├── __pycache__
│   │   ├── _bootstrap.cpython-310.pyc
│   │   └── _bootstrap.cpython-37.pyc
│   ├── train_prealign.py
│   └── train_skeleton_gircse.py
└── src
    ├── data
    │   ├── cache_manager.py
    │   ├── dataset.py
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── cache_manager.cpython-310.pyc
    │   │   ├── dataset.cpython-310.pyc
    │   │   ├── __init__.cpython-310.pyc
    │   │   └── samplers.cpython-310.pyc
    │   └── samplers.py
    ├── evaluation
    │   ├── evaluator.py
    │   ├── __init__.py
    │   └── __pycache__
    │       ├── evaluator.cpython-310.pyc
    │       └── __init__.cpython-310.pyc
    ├── __init__.py
    ├── losses
    │   ├── classwise_infonce.py
    │   ├── __init__.py
    │   ├── iterative_refinement_regularizer.py
    │   ├── __pycache__
    │   │   ├── classwise_infonce.cpython-310.pyc
    │   │   ├── __init__.cpython-310.pyc
    │   │   ├── iterative_refinement_regularizer.cpython-310.pyc
    │   │   └── stepwise_infonce.cpython-310.pyc
    │   └── stepwise_infonce.py
    ├── models
    │   ├── encoder.py
    │   ├── generative_pooling.py
    │   ├── gircse_adapter.py
    │   ├── gircse_loader.py
    │   ├── __init__.py
    │   ├── projection.py
    │   ├── __pycache__
    │   │   ├── encoder.cpython-310.pyc
    │   │   ├── generative_pooling.cpython-310.pyc
    │   │   ├── gircse_adapter.cpython-310.pyc
    │   │   ├── gircse_loader.cpython-310.pyc
    │   │   ├── __init__.cpython-310.pyc
    │   │   ├── projection.cpython-310.pyc
    │   │   ├── qformer_projector.cpython-310.pyc
    │   │   ├── skeleton_gircse.cpython-310.pyc
    │   │   ├── skeleton_prompt_builder.cpython-310.pyc
    │   │   └── soft_token_generator.cpython-310.pyc
    │   ├── qformer_projector.py
    │   ├── skeleton_gircse.py
    │   ├── skeleton_prompt_builder.py
    │   └── soft_token_generator.py
    ├── __pycache__
    │   ├── __init__.cpython-310.pyc
    │   └── __init__.cpython-37.pyc
    ├── text_branch
    │   ├── cache_text_bank.py
    │   ├── description_templates.py
    │   ├── encode_text_gircse.py
    │   ├── generate_rich_description.py
    │   ├── __init__.py
    │   └── __pycache__
    │       ├── cache_text_bank.cpython-310.pyc
    │       ├── description_templates.cpython-310.pyc
    │       ├── description_templates.cpython-37.pyc
    │       ├── encode_text_gircse.cpython-310.pyc
    │       ├── generate_rich_description.cpython-310.pyc
    │       ├── generate_rich_description.cpython-37.pyc
    │       ├── __init__.cpython-310.pyc
    │       └── __init__.cpython-37.pyc
    ├── third_party
    │   ├── gircse_embedding
    │   │   ├── base.py
    │   │   ├── __init__.py
    │   │   ├── LICENSE
    │   │   ├── model.py
    │   │   ├── README.md
    │   │   └── trainer.py
    │   ├── __init__.py
    │   ├── lavis_blip2_qformer
    │   │   ├── __init__.py
    │   │   ├── LICENSE.txt
    │   │   ├── __pycache__
    │   │   │   ├── __init__.cpython-310.pyc
    │   │   │   └── Qformer.cpython-310.pyc
    │   │   ├── Qformer.py
    │   │   └── README.md
    │   ├── __pycache__
    │   │   └── __init__.cpython-310.pyc
    │   └── shift_gcn
    │       ├── graph.py
    │       ├── __init__.py
    │       ├── LICENSE.txt
    │       ├── model.py
    │       ├── __pycache__
    │       │   ├── graph.cpython-310.pyc
    │       │   ├── __init__.cpython-310.pyc
    │       │   ├── model.cpython-310.pyc
    │       │   └── shift.cpython-310.pyc
    │       ├── README.md
    │       └── shift.py
    ├── train
    │   ├── common.py
    │   ├── factory.py
    │   ├── __init__.py
    │   └── __pycache__
    │       ├── common.cpython-310.pyc
    │       ├── common.cpython-37.pyc
    │       ├── factory.cpython-310.pyc
    │       ├── __init__.cpython-310.pyc
    │       └── __init__.cpython-37.pyc
    └── utils
        ├── checkpoint.py
        ├── config_utils.py
        ├── distributed.py
        ├── __init__.py
        ├── logging_utils.py
        ├── metrics.py
        ├── __pycache__
        │   ├── checkpoint.cpython-310.pyc
        │   ├── config_utils.cpython-310.pyc
        │   ├── distributed.cpython-310.pyc
        │   ├── __init__.cpython-310.pyc
        │   ├── logging_utils.cpython-310.pyc
        │   ├── metrics.cpython-310.pyc
        │   ├── seed.cpython-310.pyc
        │   ├── torch_utils.cpython-310.pyc
        │   └── wandb_utils.cpython-310.pyc
        ├── seed.py
        ├── torch_utils.py
        └── wandb_utils.py

31 directories, 149 files

默认 projector 是 `part_aware_qformer`，使用 7 个 query：
`head`、`left_arm`、`right_arm`、`torso`、`left_leg`、`right_leg`、`global`。
本版本只做 Part-aware Query Initialization + Global Cross-Attention，不启用 `L_part` 弱监督。
默认配置已写入服务器路径：
- Shift-GCN: `/data/chenle/GIRCSE/HAR/models`
- GIRCSE-Qwen7B: `/data/chenle/GIRCSE/GIRCSE-QWEN7B` 
- Qwen2.5-7B-Instruct: `/data/chenle/GIRCSE/Qwen2.5-7B`
`GIRCSE-Qwen7B`是LoRA adapter目录，不是完整基座模型。
`/data/chenle/GIRCSE/Qwen2.5-7B` 作为本地 base model，再挂载
`/data/chenle/GIRCSE/GIRCSE-QWEN7B` adapter，避免服务器无外网时误连 Hugging Face Hub。
Qwen2.5-7B 的 hidden size 是 3584，因此 projector `llm_dim` 和 text bank embedding
维度都配置为 3584。
本地不要求存在这些模型目录；部署到服务器后按配置运行即可。

默认数据配置使用已经预处理好的 NTU `.npz` 文件：
- NTU120: `/data/chenle/GIRCSE/HAR/data/ntu_120/NTU120.npz`
- NTU60: `/data/chenle/GIRCSE/HAR/data/ntu_60/NTU_60.npz`
`.npz` 内部应包含 `x_data` 与 `y_data`，其中 `x_data` 采用 `[N, T, M*V*C]`
布局，训练时会转换为 Shift-GCN 的 `[C, T, V, M]`。

远程服务器（实验运行环境）的Python环境：/data/chenle/.conda/envs/gircse
本地（agent修改代码）的Python环境：/Users/chenle/Desktop/gircse_env

# Instruction
下面是原始版本的项目介绍，你review并熟悉一下项目。现在的项目过于臃肿，比如configs文件太多，还有configs中有些参数都没有写清，运行指令繁杂等。我想实现用一个配置文件（通过修改配置文件的内容）即可实现所有训练（在NTU55_5、NTU48_12、NTU110_10、NTU96_24)，然后参数要全面，包括选用cuda、batch_size、eval_batch_size、eval_on_train、eval_steps等，然后将训练中的每个epoch后的权重都要保存下来（注意不用保存Qwen和GIRCSE的，它们是冻结的，只用保存训练的那些组件的权重就行），然后如果文件夹的名称一定要直观，比如train_NTU_55_5_BS5_EP5_K5这样，最好还能记录下训练起止时间到一个元信息json中（或者有更好的记录方式）。评估这边也是同理，我希望一个config解决。你先批判当前项目的问题，指出问题，然后结合上面我所说的进行修改。修改后依然以一个代码审查员的视角，审查我的代码有何不足，并进行修改，直到你觉得可以交付给你的同行Claude Opus 4.7进行审阅了。）



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

NTU60 55/5 评估使用单独配置，避免把训练配置直接当测试配置：

```text
configs/eval_zsl_ntu60_55_5.yaml
configs/eval_gzsl_ntu60_55_5.yaml
configs/eval_k_scaling_ntu60_55_5.yaml
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

Stage 2 的输出目录默认使用短命名：

```text
outputs/models/<dataset>_<split>_K<k_train>_Epoch<epochs>/
```

例如 `model.soft_tokens.k_train=10` 且 `train.epochs=10` 时，NTU60 55/5 会保存到：

```text
outputs/models/ntu60_55_5_K10_Epoch10/
```

每个 epoch 会保存一个 checkpoint：

```text
epoch_1.ckpt
epoch_2.ckpt
...
last.ckpt
```

Stage 2 checkpoint 只保存 skeleton 侧状态，即 `shift_gcn.*` 和 `token_projector.*`。
冻结的 Qwen/GIRCSE 会在运行时从本地模型路径重新加载，不会重复写入每个 epoch checkpoint。

同时 `outputs/models/all/latest_run.json` 会记录最近一次 Stage 2 运行目录、最新
epoch、loss 与 checkpoint 路径；`outputs/models/all/runs.jsonl` 会追加记录每个 epoch。

Projector 消融配置：

```bash
python scripts/train_skeleton_gircse.py --config configs/projector_linear.yaml --wandb_mode offline
python scripts/train_skeleton_gircse.py --config configs/projector_general_qformer.yaml --wandb_mode offline
python scripts/train_skeleton_gircse.py --config configs/projector_part_aware_qformer.yaml --wandb_mode offline
```

ZSL/GZSL 评估：

`eval_zsl.py` 和 `eval_gzsl.py` 使用独立 eval config 中的单个 `eval.k` 评估；
如果要测试多个 `K`，使用 `eval_k_scaling.py` 和 `eval.k_values`。

```bash
CUDA_VISIBLE_DEVICES=1,3,4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/eval_zsl.py \
  --config configs/eval_zsl_ntu60_55_5.yaml \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/ntu60_55_5_K10_Epoch10/epoch_8.ckpt \
  --wandb_mode offline

CUDA_VISIBLE_DEVICES=1,2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/eval_gzsl.py \
  --config configs/eval_gzsl_ntu60_55_5.yaml \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/skeleton_gircse-ntu60-split_55_5-modality_skeleton-loss_stepwise_infonce_irr-proj_part_aware_qformer-dim_3584-K_5-eade367866-20260519_141225/last.ckpt \
  --wandb_mode offline

CUDA_VISIBLE_DEVICES=1,2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/eval_k_scaling.py \
  --config configs/eval_k_scaling_ntu60_55_5.yaml \
  --checkpoint /data/chenle/GIRCSE/HAR/outputs/models/skeleton_gircse-ntu60-split_55_5-modality_skeleton-loss_stepwise_infonce_irr-proj_part_aware_qformer-dim_3584-K_5-eade367866-20260519_141225/last.ckpt \
  --wandb_mode offline
```



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
src/                # 核心源码
data/ntu_60/        # 预处理后的 NTU60 npz
data/ntu_120/       # 预处理后的 NTU120 npz
data/cache/         # 预处理缓存
outputs/            # checkpoint、metrics、predictions
```
