from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from latent_switch_mindspore.dataset import LatentSwitchSFTSource
from latent_switch_mindspore.records import SFTBuildConfig, build_sft_record, read_records, write_jsonl
from latent_switch_mindspore.tokens import validate_tokenizer_contract


def load_tokenizer(path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise ImportError("The CLI requires transformers to load a tokenizer.") from exc
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    validate_tokenizer_contract(tokenizer)
    return tokenizer


def cmd_build_sft(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer)
    cfg = SFTBuildConfig(
        compression_ratio_threshold=args.compression_ratio_threshold,
        max_distilled_cot_tokens=args.max_distilled_cot_tokens,
        latent_pad_token=args.latent_pad_token,
        latent_min=args.latent_min,
        latent_max=args.latent_max,
        cot_loss_weight=args.cot_loss_weight,
        answer_loss_weight=args.answer_loss_weight,
        state_align_loss_weight=args.state_align_loss_weight,
    )
    rows = read_records(args.input)
    built = []
    reasons: Counter[str] = Counter()
    for row in rows:
        record, reason = build_sft_record(row, tokenizer=tokenizer, config=cfg)
        reasons[reason] += 1
        if record is not None:
            built.append(record)
    count = write_jsonl(args.output, built)
    print(json.dumps({"written": count, "reasons": dict(reasons)}, ensure_ascii=False, indent=2))
    return 0 if count > 0 else 1


def cmd_validate_dataset(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer)
    source = LatentSwitchSFTSource.from_path(args.input, tokenizer=tokenizer, max_length=args.max_length)
    errors = []
    for index in range(len(source)):
        try:
            sample = source[index]
            if len(sample["input_ids"]) != len(sample["labels"]):
                raise ValueError("input_ids and labels length mismatch")
            if any(sample["teacher_kl_mask"][pos] for pos in sample["latent_positions"]):
                raise ValueError("teacher_kl_mask overlaps latent interior")
        except Exception as exc:  # noqa: BLE001
            errors.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            if len(errors) >= args.max_errors:
                break
    report = {"records": len(source), "errors": errors, "ok": not errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def cmd_inspect_sample(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer)
    source = LatentSwitchSFTSource.from_path(args.input, tokenizer=tokenizer, max_length=args.max_length)
    sample = source[args.index]
    report = {
        "record_id": sample["record_id"],
        "seq_len": len(sample["input_ids"]),
        "spans": sample["spans"],
        "teacher_spans": sample["teacher_spans"],
        "mask_counts": {
            "prompt_mask": int(sum(sample["prompt_mask"])),
            "latent_internal_mask": int(sum(sample["latent_internal_mask"])),
            "latent_boundary_mask": int(sum(sample["latent_boundary_mask"])),
            "cot_mask": int(sum(sample["cot_mask"])),
            "answer_mask": int(sum(sample["answer_mask"])),
            "teacher_kl_mask": int(sum(sample["teacher_kl_mask"])),
        },
        "latent_length": sample["latent_length"],
        "loss_pairs": len(sample["loss_target_positions"]),
        "teacher_kl_pairs": len(sample["teacher_kl_target_positions"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Latent-Switch MindSpore dataset utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-sft", help="Build SFT records from distilled JSONL/Parquet")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--tokenizer", required=True)
    build.add_argument("--latent-min", type=int, default=1)
    build.add_argument("--latent-max", type=int, default=128)
    build.add_argument("--latent-pad-token", default="<|endoftext|>")
    build.add_argument("--compression-ratio-threshold", type=float, default=1.0)
    build.add_argument("--max-distilled-cot-tokens", type=int, default=8192)
    build.add_argument("--cot-loss-weight", type=float, default=0.0)
    build.add_argument("--answer-loss-weight", type=float, default=1.0)
    build.add_argument("--state-align-loss-weight", type=float, default=1.0)
    build.set_defaults(func=cmd_build_sft)

    validate = subparsers.add_parser("validate-dataset", help="Validate token spans and masks")
    validate.add_argument("--input", required=True)
    validate.add_argument("--tokenizer", required=True)
    validate.add_argument("--max-length", type=int, default=4096)
    validate.add_argument("--max-errors", type=int, default=20)
    validate.set_defaults(func=cmd_validate_dataset)

    inspect = subparsers.add_parser("inspect-sample", help="Inspect one materialized sample")
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--tokenizer", required=True)
    inspect.add_argument("--index", type=int, default=0)
    inspect.add_argument("--max-length", type=int, default=4096)
    inspect.set_defaults(func=cmd_inspect_sample)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
