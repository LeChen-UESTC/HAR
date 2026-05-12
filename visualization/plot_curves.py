#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot local loss/lr curves from metrics JSONL or text log.")
    parser.add_argument("--log", required=True, help="Path to metrics.jsonl or experiment log.")
    parser.add_argument("--output", default="visualization/figures/training_curves.png")
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    log_path = Path(path)
    if log_path.suffix == ".jsonl":
        records = []
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    return parse_text_log(log_path)


def parse_text_log(path: Path) -> list[dict[str, Any]]:
    records = []
    pattern = re.compile(r"epoch=(?P<epoch>\d+).*?(?:train_loss|loss)=(?P<loss>[0-9.]+)")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                records.append(
                    {
                        "epoch": int(match.group("epoch")),
                        "train_loss": float(match.group("loss")),
                    }
                )
    return records


def main() -> None:
    args = parse_args()
    records = load_records(args.log)
    if not records:
        raise RuntimeError(f"No plottable records found in {args.log}")

    import matplotlib.pyplot as plt

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = [record.get("epoch", idx + 1) for idx, record in enumerate(records)]
    keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if key != "epoch" and isinstance(value, (int, float))
        }
    )
    keys = [key for key in keys if "loss" in key or key in {"lr", "train_top1", "val_top1", "top1"}]
    if not keys:
        raise RuntimeError("No numeric loss/lr/accuracy keys found.")

    fig, ax = plt.subplots(figsize=(10, 6))
    for key in keys:
        values = [record.get(key) for record in records]
        if all(value is None for value in values):
            continue
        ax.plot(epochs, values, marker="o", linewidth=1.5, label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
