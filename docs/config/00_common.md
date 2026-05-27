# 通用配置

每次实验最常改这些：

```yaml
project.root: /data/chenle/GIRCSE/HAR
experiment.active_split: NTU55_5
runtime.cuda_visible_devices: "0,1"
runtime.device: cuda
```

`experiment.active_split` 从 `dataset_splits` 中选择数据划分：

```text
NTU55_5, NTU48_12, NTU110_10, NTU96_24
```

只在文件位置变化时改这些路径：

```yaml
paths.qwen_instruct_model
paths.gircse_base_model
paths.gircse_adapter
paths.text_bank
paths.description_cache
dataset_splits.<split>.paths.*_npz
dataset_splits.<split>.model.shift_gcn.pretrained_path
```

`paths.text_bank` 默认带模板变量，避免不同文本配置写进同一个缓存：

```yaml
paths.text_bank: "{project_root}/data/cache/text_embeddings_ntu{text_num_classes}_zsl_{description_variant}_ktext{k_text}_{text_pooling}.pt"
```

可用模板变量：`project_root`、`active_split`、`dataset_name`、`dataset_num_classes`、`split_name`、`text_num_classes`、`description_variant`、`k_text`、`text_pooling`。

不要改 `seen_classes` / `unseen_classes`，除非你在定义新的 zero-shot split。
