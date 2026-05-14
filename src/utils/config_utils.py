from __future__ import annotations

import copy
import hashlib
import json
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


def load_config(path: str | Path) -> dict[str, Any]:
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
        base = load_config(base_path)
        config = deep_update(base, config)

    config.setdefault("_meta", {})
    config["_meta"]["config_path"] = str(config_path)
    return config


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
    dataset = get_nested(config, ["dataset", "name"], "dataset")
    modality = get_nested(config, ["model", "modality"], "skeleton")
    loss_type = get_nested(config, ["loss", "type"], "loss")
    proj_type = get_nested(config, ["model", "projector", "type"], "proj")
    proj_dim = get_nested(config, ["model", "projector", "llm_dim"], "d")
    k_train = get_nested(config, ["model", "soft_tokens", "k_train"], "k")
    stage = get_nested(config, ["train", "stage"], "run")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = config_fingerprint(
        {
            "dataset": dataset,
            "modality": modality,
            "loss": loss_type,
            "projector": get_nested(config, ["model", "projector"], {}),
            "soft_tokens": get_nested(config, ["model", "soft_tokens"], {}),
            "sampling": get_nested(config, ["dataset", "sampling_strategy"], {}),
            "preprocess_version": get_nested(config, ["dataset", "preprocess_version"], None),
        }
    )
    raw = (
        f"{stage}-{dataset}-modality_{modality}-loss_{loss_type}-"
        f"proj_{proj_type}-dim_{proj_dim}-K_{k_train}-{fp}-{stamp}"
    )
    return sanitize_name(raw)


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
