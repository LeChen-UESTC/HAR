from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


DESCRIPTION_VARIANT_SHORT_NAMES = {
    "label_only": "label",
    "label_local_motion": "label_local_motion",
    "label_local_motion_object": "label_local_motion_object",
    "full": "full",
}

PROJECTOR_TYPE_NAMES = {
    "linear",
    "linear_layernorm",
    "qformer",
    "general_qformer",
    "part_aware_qformer",
}


def _load_yaml_module():
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read YAML config files. "
            "Install dependencies from requirements.txt on the target server."
        ) from exc
    return yaml


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, materialize: bool = True) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        yaml = _load_yaml_module()
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    elif config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    base_config = config.pop("base_config", None)
    if base_config:
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        elif not base_path.exists():
            local_base_path = config_path.parent / base_path.name
            if local_base_path.exists():
                base_path = local_base_path
        base = load_config(base_path, materialize=False)
        config = deep_update(base, config)

    config.setdefault("_meta", {})
    config["_meta"]["config_path"] = str(config_path)
    if materialize:
        return normalize_config(config, config_path)
    return config


def normalize_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    result = copy.deepcopy(config)
    project_root = _project_root(result, config_path)
    result = _apply_active_split(result)
    result = _apply_named_preset(result, "train_presets", ["train", "stage"])
    result = _apply_named_preset(result, "eval_presets", ["eval", "task"])
    _normalize_aliases(result)
    result.setdefault("_meta", {})
    result["_meta"]["project_root"] = str(project_root)
    result = expand_config_templates(result)
    return result


def expand_config_templates(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    project_root = Path(
        str(get_nested(result, ["_meta", "project_root"], get_nested(result, ["project", "root"], ".")))
    )
    result.setdefault("_meta", {})
    text_variant = get_nested(result, ["text_branch", "description_variant"], "")
    projector_type = get_nested(result, ["model", "projector", "type"], "")
    result["_meta"]["text_variant"] = short_description_variant(text_variant)
    result["_meta"]["text_mode"] = text_mode_suffix_from_variant(text_variant)
    result["_meta"]["projector_type"] = normalized_projector_type(projector_type)
    result["_meta"]["projector_mode"] = projector_suffix_from_type(projector_type)
    return _expand_templates(result, _template_replacements(result, project_root))


def _project_root(config: Mapping[str, Any], config_path: Path) -> Path:
    configured = get_nested(config, ["project", "root"])
    if configured:
        expanded = str(configured).replace("{config_dir}", str(config_path.parent))
        return Path(os.path.expanduser(os.path.expandvars(expanded)))
    return config_path.parent.parent.resolve()


def _apply_active_split(config: dict[str, Any]) -> dict[str, Any]:
    splits = config.pop("dataset_splits", None)
    if not splits:
        return config

    active_split = get_nested(
        config,
        ["experiment", "active_split"],
        get_nested(config, ["dataset", "active_split"]),
    )
    if not active_split:
        raise ValueError("dataset_splits is configured but experiment.active_split is missing.")
    if active_split not in splits:
        choices = ", ".join(sorted(str(key) for key in splits))
        raise ValueError(f"Unknown active split {active_split!r}. Available splits: {choices}")

    result = deep_update(config, splits[active_split])
    result.setdefault("_meta", {})
    result["_meta"]["active_split"] = str(active_split)
    return result


def _apply_named_preset(
    config: dict[str, Any],
    presets_key: str,
    active_key_path: list[str],
) -> dict[str, Any]:
    presets = config.pop(presets_key, None)
    if not presets:
        return config

    active_name = get_nested(config, active_key_path)
    if not active_name:
        joined = ".".join(active_key_path)
        raise ValueError(f"{presets_key} is configured but {joined} is missing.")
    if active_name not in presets:
        choices = ", ".join(sorted(str(key) for key in presets))
        raise ValueError(f"Unknown {presets_key} preset {active_name!r}. Available presets: {choices}")

    result = deep_update(config, presets[active_name])
    result.setdefault("_meta", {})
    result["_meta"][presets_key.removesuffix("_presets")] = str(active_name)
    return result


def _normalize_aliases(config: dict[str, Any]) -> None:
    train_cfg = config.setdefault("train", {})
    eval_cfg = config.setdefault("eval", {})
    _copy_alias(train_cfg, "eval_on_train", "eval_during_train")
    _copy_alias(train_cfg, "eval_every_epochs", "eval_freq")
    _copy_alias(eval_cfg, "eval_batch_size", "batch_size")


def _copy_alias(config: dict[str, Any], alias: str, canonical: str) -> None:
    if alias not in config:
        return
    if canonical in config and config[canonical] != config[alias]:
        raise ValueError(
            f"Conflicting config values for {alias!r} and {canonical!r}: "
            f"{config[alias]!r} != {config[canonical]!r}"
        )
    config[canonical] = config[alias]


def _template_replacements(config: Mapping[str, Any], project_root: Path) -> dict[str, str]:
    text_num_classes = get_nested(
        config,
        ["text_branch", "generation", "num_classes"],
        get_nested(config, ["dataset", "num_classes"], ""),
    )
    description_variant = get_nested(config, ["text_branch", "description_variant"], "")
    text_variant = short_description_variant(description_variant)
    text_mode = text_mode_suffix_from_variant(description_variant)
    projector_type = normalized_projector_type(get_nested(config, ["model", "projector", "type"], ""))
    projector_mode = projector_suffix_from_type(projector_type)
    values = {
        "active_split": get_nested(
            config,
            ["_meta", "active_split"],
            get_nested(config, ["experiment", "active_split"], ""),
        ),
        "dataset_name": get_nested(config, ["dataset", "name"], ""),
        "dataset_num_classes": get_nested(config, ["dataset", "num_classes"], ""),
        "split_name": get_nested(
            config,
            ["dataset", "split_name"],
            get_nested(config, ["dataset", "split"], ""),
        ),
        "text_num_classes": text_num_classes,
        "description_variant": description_variant,
        "description_variant_short": text_variant,
        "k_text": get_nested(config, ["text_branch", "embedding", "k_text"], ""),
        "projector_mode": projector_mode,
        "projector_type": projector_type,
        "text_mode": text_mode,
        "text_variant": text_variant,
        "text_pooling": get_nested(config, ["text_branch", "embedding", "pooling"], ""),
    }
    replacements = {"project_root": str(project_root)}
    for key, value in values.items():
        if value is not None and value != "":
            if key in {"text_mode", "projector_mode"}:
                replacements[key] = str(value)
            else:
                replacements[key] = sanitize_name(str(value))
    return replacements


def short_description_variant(value: Any) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    if raw not in DESCRIPTION_VARIANT_SHORT_NAMES:
        choices = ", ".join(sorted(DESCRIPTION_VARIANT_SHORT_NAMES))
        raise ValueError(
            f"Unknown text_branch.description_variant={raw!r}. Available values: {choices}"
        )
    return DESCRIPTION_VARIANT_SHORT_NAMES[raw]


def text_mode_suffix_from_variant(value: Any) -> str:
    short = short_description_variant(value)
    return f"_{short}" if short else ""


def text_mode_suffix(config: Mapping[str, Any]) -> str:
    meta_value = get_nested(config, ["_meta", "text_mode"])
    if meta_value:
        return str(meta_value)
    return text_mode_suffix_from_variant(get_nested(config, ["text_branch", "description_variant"], ""))


def normalized_projector_type(value: Any) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    if raw not in PROJECTOR_TYPE_NAMES:
        choices = ", ".join(sorted(PROJECTOR_TYPE_NAMES))
        raise ValueError(f"Unknown model.projector.type={raw!r}. Available values: {choices}")
    return raw


def projector_suffix_from_type(value: Any) -> str:
    projector_type = normalized_projector_type(value)
    return f"_{projector_type}" if projector_type else ""


def projector_suffix(config: Mapping[str, Any]) -> str:
    meta_value = get_nested(config, ["_meta", "projector_mode"])
    if meta_value:
        return str(meta_value)
    return projector_suffix_from_type(get_nested(config, ["model", "projector", "type"], ""))


def run_identity_suffix(config: Mapping[str, Any]) -> str:
    return f"{text_mode_suffix(config)}{projector_suffix(config)}"


def _expand_templates(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {key: _expand_templates(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_templates(item, replacements) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(os.path.expanduser(value))
        for key, replacement in replacements.items():
            expanded = expanded.replace("{" + key + "}", replacement)
        return expanded
    return value


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".yaml", ".yml"}:
        yaml = _load_yaml_module()
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                to_builtin(config),
                handle,
                allow_unicode=False,
                sort_keys=False,
                default_flow_style=False,
            )
    elif output_path.suffix.lower() == ".json":
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(to_builtin(config), handle, indent=2, ensure_ascii=True)
    else:
        raise ValueError(f"Unsupported config output format: {output_path.suffix}")


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config

    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key_path, raw_value = item.split("=", 1)
        value = parse_scalar(raw_value)
        cursor = result
        parts = key_path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot override nested key through non-dict: {key_path}")
        cursor[parts[-1]] = value
    return result


def parse_scalar(raw_value: str) -> Any:
    lower = raw_value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    if "," in raw_value:
        return [parse_scalar(part.strip()) for part in raw_value.split(",")]
    return raw_value


def config_fingerprint(config: Mapping[str, Any], length: int = 10) -> str:
    payload = json.dumps(to_builtin(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def build_experiment_name(config: Mapping[str, Any]) -> str:
    if str(get_nested(config, ["experiment", "name_style"], "")).lower() == "compact":
        return build_compact_experiment_name(config)

    dataset = get_nested(config, ["dataset", "name"], "dataset")
    split_name = get_nested(
        config,
        ["dataset", "split_name"],
        get_nested(config, ["dataset", "split"], "split"),
    )
    modality = get_nested(config, ["model", "modality"], "skeleton")
    loss_type = get_nested(config, ["loss", "type"], "loss")
    proj_type = get_nested(config, ["model", "projector", "type"], "proj")
    proj_dim = get_nested(config, ["model", "projector", "llm_dim"], "d")
    stage = get_nested(config, ["eval", "stage"], get_nested(config, ["train", "stage"], "run"))
    k_value = _display_k_for_stage(config, str(stage))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = config_fingerprint(
        {
            "dataset": dataset,
            "split_name": split_name,
            "seen_classes": get_nested(config, ["dataset", "seen_classes"], []),
            "unseen_classes": get_nested(config, ["dataset", "unseen_classes"], []),
            "modality": modality,
            "loss": loss_type,
            "projector": get_nested(config, ["model", "projector"], {}),
            "soft_tokens": get_nested(config, ["model", "soft_tokens"], {}),
            "text_branch": get_nested(config, ["text_branch"], {}),
            "text_bank": get_nested(config, ["paths", "text_bank"], None),
            "train_text_bank": get_nested(config, ["train", "text_bank_path"], None),
            "eval_text_bank": get_nested(config, ["eval", "text_bank_path"], None),
            "sampling": get_nested(config, ["dataset", "sampling_strategy"], {}),
            "preprocess_version": get_nested(config, ["dataset", "preprocess_version"], None),
        }
    )
    raw = (
        f"{stage}-{dataset}-split_{split_name}-modality_{modality}-loss_{loss_type}-"
        f"proj_{proj_type}-dim_{proj_dim}-K_{k_value}-{fp}-{stamp}{run_identity_suffix(config)}"
    )
    return sanitize_name(raw)


def build_compact_experiment_name(config: Mapping[str, Any]) -> str:
    dataset = get_nested(config, ["dataset", "name"], "dataset")
    split_name = get_nested(
        config,
        ["dataset", "split_name"],
        get_nested(config, ["dataset", "split"], "split"),
    )
    run_kind = get_nested(config, ["_meta", "run_kind"], get_nested(config, ["run", "mode"], "train"))
    dataset_label = _dataset_split_label(str(dataset), str(split_name))
    if run_kind == "eval":
        task = get_nested(config, ["eval", "task"], get_nested(config, ["eval", "stage"], "eval"))
        batch_size = get_nested(
            config,
            ["eval", "eval_batch_size"],
            get_nested(config, ["eval", "batch_size"], "bs"),
        )
        k_eval = _display_k_value(
            get_nested(
                config,
                ["eval", "k"],
                get_nested(
                    config,
                    ["eval", "k_values"],
                    get_nested(config, ["model", "soft_tokens", "k_test"], "k"),
                ),
            )
        )
        return sanitize_name(f"eval_{task}_{dataset_label}_BS{batch_size}_K{k_eval}{run_identity_suffix(config)}")

    stage = str(get_nested(config, ["train", "stage"], "train"))
    batch_size = get_nested(config, ["train", "batch_size"], "bs")
    epochs = get_nested(config, ["train", "epochs"], "ep")
    if stage in {"prealign", "warmup"}:
        return sanitize_name(f"train_{stage}_{dataset_label}_BS{batch_size}_EP{epochs}{run_identity_suffix(config)}")
    k_train = _display_k_for_stage(config, stage)
    return sanitize_name(f"train_{stage}_{dataset_label}_BS{batch_size}_EP{epochs}_K{k_train}{run_identity_suffix(config)}")


def _display_k_for_stage(config: Mapping[str, Any], stage: str) -> str:
    normalized = stage.lower()
    if normalized in {"eval_zsl", "eval_gzsl", "eval_k_scaling", "zsl", "gzsl", "k_scaling"}:
        return _display_k_value(
            get_nested(
                config,
                ["eval", "k"],
                get_nested(
                    config,
                    ["eval", "k_values"],
                    get_nested(config, ["model", "soft_tokens", "k_test"], "k"),
                ),
            )
        )
    if normalized in {"prealign", "warmup"}:
        return "none"
    return _display_k_value(get_nested(config, ["model", "soft_tokens", "k_train"], "k"))


def _display_k_value(value: Any) -> str:
    if isinstance(value, list):
        return "-".join(str(item) for item in value)
    return str(value)


def ensure_text_mode_suffix(value: str, config: Mapping[str, Any]) -> str:
    return ensure_run_identity_suffix(value, config)


def ensure_run_identity_suffix(value: str, config: Mapping[str, Any]) -> str:
    name = sanitize_name(value)
    text_suffix = text_mode_suffix(config)
    projector_mode = projector_suffix(config)
    if not text_suffix and not projector_mode:
        return name
    known_text_suffixes = {
        f"_{short}" for short in DESCRIPTION_VARIANT_SHORT_NAMES.values()
    }
    known_projector_suffixes = {
        f"_{projector_type}" for projector_type in PROJECTOR_TYPE_NAMES
    }

    base = name
    projector_suffix_in_name = None
    text_suffix_in_name = None

    canonical_projector = _pop_known_suffix(name, known_projector_suffixes)
    if canonical_projector:
        base, projector_suffix_in_name = canonical_projector
        canonical_text = _pop_known_suffix(base, known_text_suffixes)
        if canonical_text:
            base, text_suffix_in_name = canonical_text
    else:
        reverse_text = _pop_known_suffix(name, known_text_suffixes)
        if reverse_text:
            base, text_suffix_in_name = reverse_text
            reverse_projector = _pop_known_suffix(base, known_projector_suffixes)
            if reverse_projector:
                base, projector_suffix_in_name = reverse_projector

    if text_suffix_in_name and text_suffix_in_name != text_suffix:
        raise ValueError(
            f"exp_name already contains text_mode {text_suffix_in_name}, "
            f"but current config requires {text_suffix}"
        )
    if projector_suffix_in_name and projector_suffix_in_name != projector_mode:
        raise ValueError(
            f"exp_name already contains projector suffix {projector_suffix_in_name}, "
            f"but current config requires {projector_mode}"
        )
    return sanitize_name(f"{base}{text_suffix}{projector_mode}")


def _pop_known_suffix(value: str, suffixes: set[str]) -> tuple[str, str] | None:
    suffix = next(
        (item for item in sorted(suffixes, key=len, reverse=True) if value.endswith(item)),
        None,
    )
    if not suffix:
        return None
    return value[: -len(suffix)], suffix


def _dataset_split_label(dataset: str, split_name: str) -> str:
    normalized = dataset.lower()
    if normalized in {"ntu60", "ntu120"}:
        return f"NTU_{split_name}"
    return f"{dataset}_{split_name}"


def get_nested(config: Mapping[str, Any], keys: list[str], default: Any = None) -> Any:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def to_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:220]


def prepare_run_dirs(config: Mapping[str, Any], exp_name: str | None = None) -> dict[str, Path]:
    name = ensure_text_mode_suffix(exp_name or build_experiment_name(config), config)
    output_root = Path(str(get_nested(config, ["experiment", "output_root"], "outputs")))
    log_root = Path(str(get_nested(config, ["experiment", "log_root"], "logs")))
    model_dir = output_root / "models" / name
    eval_dir = output_root / "eval" / name
    run_kind = str(get_nested(config, ["_meta", "run_kind"], "train"))
    log_root.mkdir(parents=True, exist_ok=True)
    if run_kind == "eval":
        eval_dir.mkdir(parents=True, exist_ok=True)
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
    return {
        "exp_name": Path(name),
        "model_dir": model_dir,
        "eval_dir": eval_dir,
        "log_root": log_root,
    }
