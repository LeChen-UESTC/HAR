# 结构化文本描述

当前不使用生成模型。`paths.description_cache` 必须指向已经准备好的结构化 JSON，例如：

```yaml
paths.description_cache: "{project_root}/data/cache/descriptions.json"
```

常改字段：

```yaml
paths.description_cache
text_branch.num_classes
text_branch.description_variant
```

`description_variant` 可选：

```text
structured
```

每个类别只允许这 4 个字段：

```json
{
  "label": "writing",
  "observable_motion": "One hand performs small repeated strokes in front of the upper body.",
  "key_body_parts": ["hand", "arm", "torso"],
  "temporal_phases": {
    "start": "The hand moves toward the front body area.",
    "middle": "The hand performs repeated small writing-like strokes.",
    "end": "The hand movement slows down or stops."
  }
}
```

说明：

- `text_branch.num_classes` 控制从类别表和描述文件中读取多少类。
- `temporal_phases` 必须且只能包含 `start`、`middle`、`end`。
- 不允许额外字段；object / environment / context 不进入当前文本原型。
- 缓存 text bank 时会逐类校验 schema，缺类、字段缺失、label 不一致都会直接报错。
