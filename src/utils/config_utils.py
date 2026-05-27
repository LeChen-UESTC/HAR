from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


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
    result = _expand_templates(result, {"project_root": str(project_root)})
    result.setdefault("_meta", {})
    result["_meta"]["project_root"] = str(project_root)
    return result


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
    k_train = _display_k_value(get_nested(config, ["model", "soft_tokens", "k_train"], "k"))
    stage = get_nested(config, ["eval", "stage"], get_nested(config, ["train", "stage"], "run"))
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
            "sampling": get_nested(config, ["dataset", "sampling_strategy"], {}),
            "preprocess_version": get_nested(config, ["dataset", "preprocess_version"], None),
        }
    )
    raw = (
        f"{stage}-{dataset}-split_{split_name}-modality_{modality}-loss_{loss_type}-"
        f"proj_{proj_type}-dim_{proj_dim}-K_{k_train}-{fp}-{stamp}"
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
        return sanitize_name(f"eval_{task}_{dataset_label}_BS{batch_size}_K{k_eval}")

    batch_size = get_nested(config, ["train", "batch_size"], "bs")
    epochs = get_nested(config, ["train", "epochs"], "ep")
    k_train = _display_k_value(get_nested(config, ["model", "soft_tokens", "k_train"], "k"))
    return sanitize_name(f"train_{dataset_label}_BS{batch_size}_EP{epochs}_K{k_train}")


def _display_k_value(value: Any) -> str:
    if isinstance(value, list):
        return "-".join(str(item) for item in value)
    return str(value)


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
    name = exp_name or build_experiment_name(config)
    output_root = Path(str(get_nested(config, ["experiment", "output_root"], "outputs")))
    log_root = Path(str(get_nested(config, ["experiment", "log_root"], "logs")))
    model_dir = output_root / "models" / name
    eval_dir = output_root / "eval" / name
    log_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    return {
        "exp_name": Path(name),
        "model_dir": model_dir,
        "eval_dir": eval_dir,
        "log_root": log_root,
    }
