# 生成结构化文本描述

命令：

```bash
python scripts/generate_rich_description.py --config configs/har_experiment.yaml
```

常改字段：

```yaml
paths.description_cache
text_branch.generation.num_classes
text_branch.generation.max_new_tokens
text_branch.generation.temperature
text_branch.generation.top_p
text_branch.generation.max_retries
text_branch.generation.dry_run
```

说明：

- `num_classes: 120` 表示生成 120 个动作类别描述。
- `dry_run: true` 只写启发式结构化描述，用来测试流程。
- 每个类别只允许这 4 个字段：`label`、`observable_motion`、`key_body_parts`、`temporal_phases`。
- `temporal_phases` 必须且只能包含 `start`、`middle`、`end`。
- 改 prompt 逻辑、模型路径、类别数或生成参数后，应重新生成 `description_cache`。
