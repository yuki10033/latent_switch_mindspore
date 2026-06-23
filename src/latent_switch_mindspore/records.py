from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from latent_switch_mindspore.tokens import (
    DEFAULT_LATENT_PAD_TOKEN,
    LATENT_THINK_END,
    LATENT_THINK_START,
    THINK_END,
    THINK_START,
)


MAX_ALLOWED_LATENT_STEPS = 256


@dataclass(frozen=True)
class SFTBuildConfig:
    compression_ratio_threshold: float = 1.0
    max_distilled_cot_tokens: int = 8192
    latent_pad_token: str = DEFAULT_LATENT_PAD_TOKEN
    latent_min: int = 1
    latent_max: int = 128
    cot_loss_weight: float = 0.0
    answer_loss_weight: float = 1.0
    state_align_loss_weight: float = 1.0
    latent_start_ce_loss_weight: float = 1.0
    latent_end_ce_loss_weight: float = 1.0
    im_end_ce_loss_weight: float = 1.0


def compact_text(text: Any) -> str:
    return str(text or "").strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def normalize_selected_insight_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return normalize_space(" ".join(str(item).strip() for item in raw if str(item).strip()))
    text = str(raw).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return normalize_space(" ".join(str(item).strip() for item in parsed if str(item).strip()))
        except Exception:
            pass
    return normalize_space(text)


def replace_user_intuition_phrases(text: str) -> str:
    pattern = re.compile(r"\b(?:the\s+)?user['’]s intuition\b", flags=re.IGNORECASE)

    def repl(match: re.Match[str]) -> str:
        matched = match.group(0)
        return "My intuition" if matched and matched[0].isupper() else "my intuition"

    return pattern.sub(repl, text or "")


def difficulty_bucket(is_correct: bool, distilled_cot_tokens: int) -> Tuple[str, int]:
    if not is_correct:
        return "hard", 2
    if distilled_cot_tokens < 500:
        return "easy", 0
    if distilled_cot_tokens <= 8192:
        return "medium", 1
    return "hard", 2


def _nested(row: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = row.get(key, {})
    return value if isinstance(value, dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        text = compact_text(value)
        if text:
            return text
    return ""


def _token_count(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_sft_record(
    row: Dict[str, Any],
    tokenizer: Any,
    config: Optional[SFTBuildConfig] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Convert one distilled row into a latent-switch SFT record."""
    cfg = config or SFTBuildConfig()
    stage1 = _nested(row, "stage1")
    stage2 = _nested(row, "stage2")
    token_stats = _nested(row, "token_stats")
    validation = _nested(stage2, "validation")

    ratio = safe_float(
        token_stats.get("compression_ratio_vs_primary_output", row.get("compression_ratio_vs_primary_output")),
        default=0.0,
    )
    if ratio and ratio >= cfg.compression_ratio_threshold:
        return None, "filtered_ratio"

    question = _first_text(row.get("question"), row.get("problem"), row.get("prompt"))
    if not question:
        return None, "missing_question"

    distilled_cot = _first_text(stage2.get("distilled_cot"), row.get("distilled_cot"), row.get("assistant_cot"))
    distilled_cot = replace_user_intuition_phrases(distilled_cot)
    if not distilled_cot:
        return None, "missing_cot"

    answer_text = _first_text(
        stage2.get("answer"),
        row.get("answer"),
        row.get("assistant_answer"),
        row.get("ground_truth"),
    )
    if not answer_text:
        return None, "missing_answer"

    correct_insight = _first_text(
        stage1.get("correct_insight"),
        row.get("correct_insight"),
        row.get("solution_intuition"),
    )
    selected_insight_text = normalize_selected_insight_text(
        stage2.get("selected_insight_text", row.get("selected_insight_text"))
    )
    insight_for_latent = correct_insight or selected_insight_text
    insight_token_len = _token_count(tokenizer, insight_for_latent)

    latent_max = min(int(cfg.latent_max), MAX_ALLOWED_LATENT_STEPS) if cfg.latent_max > 0 else MAX_ALLOWED_LATENT_STEPS
    n_latent_steps = max(int(cfg.latent_min), insight_token_len // 2)
    n_latent_steps = min(n_latent_steps, latent_max)
    if n_latent_steps > MAX_ALLOWED_LATENT_STEPS:
        return None, "filtered_latent_steps_gt_256"

    distilled_cot_tokens = safe_int(token_stats.get("distilled_cot_tokens", row.get("distilled_cot_tokens")), 0)
    if distilled_cot_tokens <= 0:
        distilled_cot_tokens = _token_count(tokenizer, distilled_cot)
    if distilled_cot_tokens >= cfg.max_distilled_cot_tokens:
        return None, "filtered_distilled_cot_tokens_ge_threshold"

    is_correct = bool(validation.get("is_correct", row.get("stage2_is_correct", False)))
    difficulty, difficulty_rank = difficulty_bucket(is_correct, distilled_cot_tokens)
    latent_placeholder = str(cfg.latent_pad_token) * int(n_latent_steps)
    assistant_content = (
        f"{LATENT_THINK_START}{latent_placeholder}{LATENT_THINK_END}"
        f"{THINK_START}{distilled_cot}{THINK_END}{answer_text}"
    )

    selected_prompt = question
    if selected_insight_text:
        selected_prompt = (
            "Solve the following problem by continuing from my intuition.\n\n"
            "My intuition may be correct or incorrect. Continue from it and finish the solution.\n\n"
            f"Problem:\n{question}\n\nMy Intuition:\n{selected_insight_text}"
        )

    uid = _first_text(row.get("uid"), row.get("record_id"), row.get("source_uid"))
    if not uid:
        uid_seed = f"{question}||{answer_text}||{distilled_cot[:200]}"
        uid = hashlib.md5(uid_seed.encode("utf-8")).hexdigest()

    record = {
        "record_id": uid,
        "source_uid": _first_text(row.get("source_uid"), uid),
        "question": question,
        "ground_truth": answer_text,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ],
        "assistant_cot": distilled_cot,
        "assistant_answer": answer_text,
        "difficulty": difficulty,
        "difficulty_rank": difficulty_rank,
        "n_latent_steps": int(n_latent_steps),
        "insight_token_len": int(insight_token_len),
        "correct_insight": correct_insight,
        "selected_insight_text": selected_insight_text,
        "compression_ratio_vs_primary_output": float(ratio),
        "distilled_cot_tokens": int(distilled_cot_tokens),
        "stage2_is_correct": is_correct,
        "latent_pad_token": str(cfg.latent_pad_token),
        "latent_loss_weight": 0.0,
        "cot_loss_weight": float(cfg.cot_loss_weight),
        "answer_loss_weight": float(cfg.answer_loss_weight),
        "latent_start_ce_loss_weight": float(cfg.latent_start_ce_loss_weight),
        "latent_end_ce_loss_weight": float(cfg.latent_end_ce_loss_weight),
        "im_end_ce_loss_weight": float(cfg.im_end_ce_loss_weight),
        "mask_prompt_loss": True,
        "mask_system_loss": True,
        "latent_backprop_strategy": "markov_state_only",
        "state_align_enabled": True,
        "state_align_loss_weight": float(cfg.state_align_loss_weight),
        "state_align_reference_messages": [
            {"role": "user", "content": selected_prompt},
            {"role": "assistant", "content": f"{THINK_START}{distilled_cot}{THINK_END}{answer_text}"},
        ],
        "state_align_target": "assistant_cot_start_state",
        "curriculum_sort_key": [int(difficulty_rank), int(n_latent_steps), int(distilled_cot_tokens)],
        "dataset_source": _first_text(row.get("dataset_source"), row.get("source")),
        "original_dataset": _first_text(row.get("original_dataset"), row.get("dataset")),
    }
    return record, "ok"


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_records(path: str | Path) -> List[Dict[str, Any]]:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        if suffix == ".json":
            payload = json.loads(path_obj.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
            if isinstance(payload, dict):
                return [payload]
            return []
        return read_jsonl(path_obj)
    if suffix == ".parquet":
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError("Reading parquet requires pandas and pyarrow.") from exc
        return pd.read_parquet(path_obj).to_dict(orient="records")
    raise ValueError(f"Unsupported data format: {path_obj}")
