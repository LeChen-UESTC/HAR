#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL_PATH = "/data/chenle/GIRCSE/Qwen2.5-7B"
DEFAULT_ADAPTER_PATH = "/data/chenle/GIRCSE/GIRCSE-QWEN7B"
DEFAULT_TEXT = "Why is it so hard to track down this card?"
DEFAULT_INSTRUCTIONS = [
    "Represent the intention of this text.",
    "Represent the emotion of this text.",
]
DEFAULT_INSTRUCTION_NAMES = ["intention", "emotion"]
DEFAULT_OUTPUT_JSON = "/data/chenle/GIRCSE/HAR/visualization/examples/gircse_soft_tokens_table4.json"
DEFAULT_GPU_IDS = "0,1,2,3"


def apply_gpu_visibility_from_argv() -> None:
    gpu_ids = DEFAULT_GPU_IDS
    if "--gpu_ids" in sys.argv:
        idx = sys.argv.index("--gpu_ids")
        if idx + 1 < len(sys.argv):
            gpu_ids = sys.argv[idx + 1]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_ids)


apply_gpu_visibility_from_argv()

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "by", "from", "as", "at",
    "this", "that", "these", "those", "it", "its", "you", "your",
    "i", "we", "they", "he", "she", "them", "us",
    "what", "why", "how", "when", "where", "which", "who",
    "can", "could", "would", "should", "will", "may", "might",
    "do", "does", "did", "not", "no", "yes",
    "text", "represent", "represents", "representation",
    "query", "instruct", "instruction", "response", "answer",
    "using", "use", "used", "based", "show", "showing",
    "before", "after", "below", "method", "system",
}

DEFAULT_JUNK_TOKENS = {
    # Common Qwen/GIRCSE tail artifacts observed after high-probability anchor
    # tokens dominate the distribution. The list is explicit so qualitative
    # filtering remains auditable.
    "acje", "addcriterion", "akest", "andalso", "atego", "backpage",
    "bel", "belgi", "bilt", "br", "buster", "cardcontent", "comings",
    "derp", "ebin", "elts", "engl", "everton", "eventqueue", "fkk",
    "generationstrategy", "glish", "ksz", "ltd", "maneu", "newcom",
    "nuest", "ocard", "ocre", "pupper", "rar", "rlen", "rtos", "safeg",
    "scii", "soles", "stdlib", "tottenham", "twor", "uckland", "unet",
    "zzle",
}

ANCHOR_LIKE_TYPES = {"anchor", "artifact", "empty", "fragment", "junk", "long", "number"}

CANONICAL_ALIASES = {
    "tracking": "track",
    "tracked": "track",
    "tracks": "track",
    "tracker": "track",
    "trackers": "track",
    "cards": "card",
    "harder": "hard",
    "hardest": "hard",
    "difficulties": "difficult",
    "frustration": "frustrating",
    "frustrated": "frustrating",
    "struggling": "struggle",
    "struggled": "struggle",
    "seeking": "seek",
    "sought": "seek",
    "questions": "question",
    "inquiries": "inquiry",
    "challenged": "challenging",
    "challenges": "challenging",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GIRCSE soft-token top-k distributions for visualization."
    )
    parser.add_argument("--base_model_path", default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--instruction", action="append", default=None)
    parser.add_argument("--instruction_name", action="append", default=None)
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--raw_topk", type=int, default=500)
    parser.add_argument("--logit_temperature", type=float, default=1.0)
    parser.add_argument("--groups", default="1-5,6-10,11-20")
    parser.add_argument("--semantic_min_len", type=int, default=3)
    parser.add_argument(
        "--junk_token",
        action="append",
        default=None,
        help="Additional normalized token to exclude from semantic/residual views.",
    )
    parser.add_argument("--gpu_ids", default=DEFAULT_GPU_IDS, help="Optional CUDA_VISIBLE_DEVICES value, e.g. 0,1,2,3.")
    parser.add_argument("--include_base", action=argparse.BooleanOptionalAction, default=True, help="Also export before-FT base model.")
    parser.add_argument("--add_eos", action="store_true")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    args = parser.parse_args()
    if args.instruction is None:
        args.instruction = list(DEFAULT_INSTRUCTIONS)
    if args.instruction_name is None:
        args.instruction_name = list(DEFAULT_INSTRUCTION_NAMES)
    return args


def parse_groups(raw: str) -> list[tuple[int, int]]:
    groups = []
    for item in raw.split(","):
        start, end = item.split("-", 1)
        groups.append((int(start), int(end)))
    return groups


def build_prompt(instruction: str, text: str) -> str:
    return f"Instruct: {instruction}\nQuery:{text}"


def load_tokenizer(base_model_path: str, add_eos: bool) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        add_eos_token=add_eos,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(base_model_path: str, attn_implementation: str | None) -> Any:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
    model.gradient_checkpointing_enable()
    model.eval()
    return model


def attach_adapter(base_model: Any, adapter_path: str) -> Any:
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map="auto",
    )
    model.eval()
    return model


def tokenize_text(tokenizer: Any, text_list: list[str]) -> Any:
    return tokenizer(
        text_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
        pad_to_multiple_of=8,
        return_token_type_ids=False,
        add_special_tokens=False,
    ).to("cuda")


def decode_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    piece = tokenizer.convert_ids_to_tokens(int(token_id))
    text = text.replace("\n", "\\n")
    return text if text.strip() else piece


def normalize_token(token: str) -> str:
    token = token.replace("Ġ", " ").replace("▁", " ")
    token = token.replace("\\n", " ")
    token = token.strip().lower()
    token = token.strip(".,;:!?\"'`“”‘’()[]{}<>")
    return token


def canonicalize(token: str) -> str:
    normalized = normalize_token(token)
    return CANONICAL_ALIASES.get(normalized, normalized)


def classify_token(token: str, stopwords: set[str], junk_tokens: set[str], semantic_min_len: int) -> str:
    normalized = normalize_token(token)
    if not normalized:
        return "empty"
    if normalized in stopwords:
        return "stopword"
    if normalized in junk_tokens:
        return "junk"
    if len(normalized) < semantic_min_len:
        return "fragment"
    if len(normalized) > 30:
        return "long"
    if re.fullmatch(r"[0-9]+", normalized):
        return "number"
    if any(
        fragment in normalized
        for fragment in [
            "/", "\\", "<", ">", "{", "}", "[", "]", "|", "&&",
            "://", ".com", ".org", ".net", "http", "www",
            "compatible", "gate", "instruction",
        ]
    ):
        return "artifact"
    if re.fullmatch(r"[a-zA-Z][a-zA-Z'-]*", normalized):
        if re.fullmatch(r"[a-z]+", normalized) and not re.search(r"[aeiouy]", normalized):
            return "junk"
        return "semantic"
    return "anchor"


@torch.no_grad()
def collect_soft_tokens(
    model: Any,
    tokenizer: Any,
    prompt: str,
    k: int,
    raw_topk: int,
    logit_temperature: float,
    stopwords: set[str],
    junk_tokens: set[str],
    semantic_min_len: int,
) -> list[dict[str, Any]]:
    inputs = tokenize_text(tokenizer, [prompt])
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    embedding_layer = model.get_input_embeddings()
    embedding_weight = embedding_layer.weight
    input_embeds = embedding_layer(input_ids)

    past_key_values = None
    use_cache = False
    records = []

    for loop_idx in range(k + 1):
        model_inputs = {
            "inputs_embeds": input_embeds if past_key_values is None else input_embeds[:, -1:, :],
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": use_cache,
            "return_dict": True,
        }
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

        outputs = model(**model_inputs)
        logits = outputs.logits[:, -1, :].float()
        probs = F.softmax(logits / logit_temperature, dim=-1)

        if loop_idx < k:
            step = loop_idx + 1
            top_probs, top_ids = torch.topk(probs[0], k=raw_topk)
            items = []
            for rank, (prob, token_id) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
                token = decode_token(tokenizer, int(token_id))
                token_type = classify_token(token, stopwords, junk_tokens, semantic_min_len)
                items.append(
                    {
                        "step": step,
                        "rank": rank,
                        "token_id": int(token_id),
                        "token": token,
                        "normalized": normalize_token(token),
                        "canonical": canonicalize(token),
                        "prob": float(prob),
                        "residual_prob": 0.0,
                        "type": token_type,
                        "keep": token_type == "semantic",
                    }
                )
            semantic_mass_topk = sum(item["prob"] for item in items if item["keep"])
            anchor_like_mass_topk = sum(item["prob"] for item in items if item["type"] in ANCHOR_LIKE_TYPES)
            raw_topk_mass = sum(item["prob"] for item in items)
            for item in items:
                if item["keep"] and semantic_mass_topk > 0:
                    item["residual_prob"] = item["prob"] / semantic_mass_topk
            records.append(
                {
                    "step": step,
                    "entropy": float(-(probs[0] * torch.log(probs[0] + 1e-12)).sum().item()),
                    "max_prob": float(top_probs[0].item()),
                    "raw_topk_mass": float(raw_topk_mass),
                    "semantic_mass_topk": float(semantic_mass_topk),
                    "anchor_like_mass_topk": float(anchor_like_mass_topk),
                    "items": items,
                    "raw_top": items[:30],
                }
            )

        soft_token_embed = (probs.to(embedding_weight.dtype) @ embedding_weight).unsqueeze(1)
        input_embeds = torch.cat([input_embeds, soft_token_embed], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.size(0), 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=-1,
        )
        past_key_values = outputs.past_key_values if use_cache else None

    return records


def filtered_top_for_step(
    record: dict[str, Any],
    topn: int,
    prob_field: str,
    prob_kind: str,
) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for item in record["items"]:
        if not item["keep"]:
            continue
        key = item["canonical"]
        if key in seen:
            continue
        seen.add(key)
        output_item = dict(item)
        output_item["raw_prob"] = item["prob"]
        output_item["prob"] = item[prob_field]
        output_item["prob_kind"] = prob_kind
        selected.append(output_item)
        if len(selected) >= topn:
            break
    return selected


def semantic_top_for_step(record: dict[str, Any], topn: int) -> list[dict[str, Any]]:
    return filtered_top_for_step(record, topn, "prob", "prob")


def residual_top_for_step(record: dict[str, Any], topn: int) -> list[dict[str, Any]]:
    return filtered_top_for_step(record, topn, "residual_prob", "residual_prob")


def group_sort_key(item: dict[str, Any], prob_field: str) -> tuple[Any, ...]:
    return (
        -item["freq"],
        -item[prob_field],
        item["best_rank"],
        item["first_step"],
        item["token"],
    )


def aggregate_group(
    records: list[dict[str, Any]],
    start: int,
    end: int,
    topn: int,
    prob_field: str = "prob_sum",
    prob_kind: str = "prob_sum",
) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "freq": 0,
            "prob_sum": 0.0,
            "residual_prob_sum": 0.0,
            "best_rank": 10**9,
            "first_step": 10**9,
            "surface_forms": set(),
            "token_ids": set(),
        }
    )

    for record in records:
        step = record["step"]
        if not (start <= step <= end):
            continue
        seen_this_step = set()
        for item in record["items"]:
            if not item["keep"]:
                continue
            key = item["canonical"]
            if key in seen_this_step:
                continue
            seen_this_step.add(key)

            stats[key]["freq"] += 1
            stats[key]["prob_sum"] += item["prob"]
            stats[key]["residual_prob_sum"] += item["residual_prob"]
            stats[key]["best_rank"] = min(stats[key]["best_rank"], item["rank"])
            stats[key]["first_step"] = min(stats[key]["first_step"], step)
            stats[key]["surface_forms"].add(item["normalized"])
            stats[key]["token_ids"].add(item["token_id"])

    grouped_items = []
    for key, value in stats.items():
        grouped_items.append(
            {
                "token": key,
                "freq": value["freq"],
                "prob_sum": value["prob_sum"],
                "residual_prob_sum": value["residual_prob_sum"],
                "best_rank": value["best_rank"],
                "first_step": value["first_step"],
                "surface_forms": sorted(value["surface_forms"]),
                "token_ids": sorted(value["token_ids"]),
            }
        )

    grouped_items.sort(key=lambda item: group_sort_key(item, prob_field))
    output = []
    for item in grouped_items[:topn]:
        output_item = dict(item)
        output_item["raw_prob_sum"] = item["prob_sum"]
        output_item["prob_sum"] = item[prob_field]
        output_item["prob_kind"] = prob_kind
        output.append(output_item)
    return output


def aggregate_group_raw(
    records: list[dict[str, Any]],
    start: int,
    end: int,
    topn: int,
) -> list[dict[str, Any]]:
    stats: dict[tuple[int, str], dict[str, Any]] = defaultdict(
        lambda: {
            "freq": 0,
            "prob_sum": 0.0,
            "best_rank": 10**9,
            "first_step": 10**9,
            "token": "",
            "normalized": "",
            "canonical": "",
            "type": "anchor",
            "keep": False,
            "token_id": None,
        }
    )

    for record in records:
        step = record["step"]
        if not (start <= step <= end):
            continue
        seen_this_step = set()
        for item in record["items"]:
            key = (item["token_id"], item["canonical"])
            if key in seen_this_step:
                continue
            seen_this_step.add(key)
            stats[key]["freq"] += 1
            stats[key]["prob_sum"] += item["prob"]
            stats[key]["best_rank"] = min(stats[key]["best_rank"], item["rank"])
            stats[key]["first_step"] = min(stats[key]["first_step"], step)
            stats[key]["token"] = item["token"]
            stats[key]["normalized"] = item["normalized"]
            stats[key]["canonical"] = item["canonical"]
            stats[key]["type"] = item["type"]
            stats[key]["keep"] = item["keep"]
            stats[key]["token_id"] = item["token_id"]

    grouped_items = list(stats.values())
    grouped_items.sort(key=lambda item: group_sort_key(item, "prob_sum"))
    return [
        {
            "token": item["token"],
            "normalized": item["normalized"],
            "canonical": item["canonical"],
            "type": item["type"],
            "keep": item["keep"],
            "token_id": item["token_id"],
            "rank": item["best_rank"],
            "prob": item["prob_sum"],
            "prob_sum": item["prob_sum"],
            "prob_kind": "prob_sum",
            "freq": item["freq"],
            "first_step": item["first_step"],
        }
        for item in grouped_items[:topn]
    ]


def build_prompt_payload(
    model_label: str,
    model: Any,
    tokenizer: Any,
    text: str,
    instruction_name: str,
    instruction: str,
    args: argparse.Namespace,
    groups: list[tuple[int, int]],
    stopwords: set[str],
) -> dict[str, Any]:
    prompt = build_prompt(instruction, text)
    records = collect_soft_tokens(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        k=args.k,
        raw_topk=args.raw_topk,
        logit_temperature=args.logit_temperature,
        stopwords=stopwords,
        junk_tokens=set(DEFAULT_JUNK_TOKENS).union(args.junk_token or []),
        semantic_min_len=args.semantic_min_len,
    )

    steps = []
    for record in records:
        steps.append(
            {
                "step": record["step"],
                "entropy": record["entropy"],
                "max_prob": record["max_prob"],
                "raw_topk_mass": record["raw_topk_mass"],
                "semantic_mass_topk": record["semantic_mass_topk"],
                "anchor_like_mass_topk": record["anchor_like_mass_topk"],
                "raw_top": record["raw_top"][: args.topk],
                "semantic_top": semantic_top_for_step(record, args.topk),
                "residual_top": residual_top_for_step(record, args.topk),
            }
        )

    return {
        "model_label": model_label,
        "instruction_name": instruction_name,
        "instruction": instruction,
        "prompt": prompt,
        "steps": steps,
        "groups": [
            {
                "name": f"{start}-{end}",
                "start": start,
                "end": end,
                "raw_top": aggregate_group_raw(records, start, end, args.topk),
                "semantic_top": aggregate_group(records, start, end, args.topk, "prob_sum", "prob_sum"),
                "residual_top": aggregate_group(records, start, end, args.topk, "residual_prob_sum", "residual_prob_sum"),
            }
            for start, end in groups
        ],
    }


def main() -> None:
    args = parse_args()
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    instruction_names = args.instruction_name
    if instruction_names is None:
        instruction_names = [f"instruction_{idx + 1}" for idx in range(len(args.instruction))]
    if len(instruction_names) != len(args.instruction):
        raise ValueError("--instruction_name must be provided once per --instruction")

    groups = parse_groups(args.groups)
    stopwords = set(DEFAULT_STOPWORDS)
    tokenizer = load_tokenizer(args.base_model_path, add_eos=args.add_eos)

    output: dict[str, Any] = {
        "metadata": {
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path,
            "text": args.text,
            "k": args.k,
            "topk": args.topk,
            "raw_topk": args.raw_topk,
            "logit_temperature": args.logit_temperature,
            "semantic_min_len": args.semantic_min_len,
            "junk_tokens": sorted(set(DEFAULT_JUNK_TOKENS).union(args.junk_token or [])),
            "groups": [{"start": start, "end": end} for start, end in groups],
        },
        "runs": [],
    }

    base_model = load_base_model(args.base_model_path, args.attn_implementation)
    if args.include_base:
        for name, instruction in zip(instruction_names, args.instruction):
            output["runs"].append(
                build_prompt_payload(
                    model_label="Before FT / Base",
                    model=base_model,
                    tokenizer=tokenizer,
                    text=args.text,
                    instruction_name=name,
                    instruction=instruction,
                    args=args,
                    groups=groups,
                    stopwords=stopwords,
                )
            )

    if args.adapter_path:
        adapted_model = attach_adapter(base_model, args.adapter_path)
        for name, instruction in zip(instruction_names, args.instruction):
            output["runs"].append(
                build_prompt_payload(
                    model_label="After FT / GIRCSE",
                    model=adapted_model,
                    tokenizer=tokenizer,
                    text=args.text,
                    instruction_name=name,
                    instruction=instruction,
                    args=args,
                    groups=groups,
                    stopwords=stopwords,
                )
            )
        del adapted_model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(output_path)


if __name__ == "__main__":
    main()
