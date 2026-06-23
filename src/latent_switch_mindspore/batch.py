from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


SEQUENCE_INT_COLUMNS = (
    "input_ids",
    "labels",
    "attention_mask",
    "position_ids",
)
SEQUENCE_FLOAT_COLUMNS = ("loss_weights",)
SEQUENCE_BOOL_COLUMNS = (
    "prompt_mask",
    "latent_internal_mask",
    "latent_boundary_mask",
    "cot_mask",
    "answer_mask",
    "teacher_kl_mask",
    "valid_token_mask",
    "latent_pad_mask",
)
BATCH_COLUMNS = (
    "input_ids",
    "labels",
    "loss_weights",
    "attention_mask",
    "position_ids",
    "prompt_mask",
    "latent_internal_mask",
    "latent_boundary_mask",
    "cot_mask",
    "answer_mask",
    "teacher_kl_mask",
    "valid_token_mask",
    "latent_pad_mask",
    "teacher_target_start",
    "latent_positions",
    "latent_slot_mask",
    "latent_lengths",
    "latent_start_positions",
    "latent_end_positions",
    "loss_source_positions",
    "loss_target_positions",
    "loss_pair_mask",
    "teacher_kl_source_positions",
    "teacher_kl_target_positions",
    "teacher_kl_pair_mask",
    "cot_branch_weight",
)


def build_effective_token_mappings(
    valid_token_mask: Sequence[bool],
    labels: Sequence[int],
    teacher_kl_mask: Sequence[bool],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    valid_positions = [idx for idx, is_valid in enumerate(valid_token_mask) if bool(is_valid)]
    loss_source_positions: List[int] = []
    loss_target_positions: List[int] = []
    kl_source_positions: List[int] = []
    kl_target_positions: List[int] = []

    for pair_index in range(1, len(valid_positions)):
        source_pos = int(valid_positions[pair_index - 1])
        target_pos = int(valid_positions[pair_index])
        if int(labels[target_pos]) != -100:
            loss_source_positions.append(source_pos)
            loss_target_positions.append(target_pos)
        if bool(teacher_kl_mask[source_pos]):
            kl_source_positions.append(source_pos)
            kl_target_positions.append(target_pos)

    return loss_source_positions, loss_target_positions, kl_source_positions, kl_target_positions


def _pad_2d(
    rows: Sequence[Sequence[Any]],
    pad_value: Any,
    dtype: np.dtype | type,
    width: int | None = None,
) -> np.ndarray:
    batch_size = len(rows)
    max_len = width if width is not None else max((len(row) for row in rows), default=0)
    output = np.full((batch_size, max_len), pad_value, dtype=dtype)
    for row_idx, row in enumerate(rows):
        values = list(row)
        if values:
            output[row_idx, : len(values)] = np.asarray(values, dtype=dtype)
    return output


def collate_samples(samples: Sequence[Dict[str, Any]], pad_token_id: int) -> Dict[str, np.ndarray]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    max_seq_len = max(len(sample["input_ids"]) for sample in samples)
    result: Dict[str, np.ndarray] = {}

    result["input_ids"] = _pad_2d([s["input_ids"] for s in samples], pad_token_id, np.int64, max_seq_len)
    result["labels"] = _pad_2d([s["labels"] for s in samples], -100, np.int64, max_seq_len)
    result["loss_weights"] = _pad_2d([s["loss_weights"] for s in samples], 0.0, np.float32, max_seq_len)

    for name in SEQUENCE_BOOL_COLUMNS:
        result[name] = _pad_2d([s[name] for s in samples], False, np.bool_, max_seq_len)
    result["attention_mask"] = result["valid_token_mask"].astype(np.int64)
    position_ids = np.zeros((len(samples), max_seq_len), dtype=np.int64)
    for row_idx, sample in enumerate(samples):
        seq_len = len(sample["input_ids"])
        position_ids[row_idx, :seq_len] = np.arange(seq_len, dtype=np.int64)
    result["position_ids"] = position_ids

    result["teacher_target_start"] = np.asarray([s["teacher_target_start"] for s in samples], dtype=np.int64)
    result["latent_lengths"] = np.asarray([s["latent_length"] for s in samples], dtype=np.int64)
    result["latent_start_positions"] = np.asarray([s["spans"]["latent_start"] for s in samples], dtype=np.int64)
    result["latent_end_positions"] = np.asarray([s["spans"]["latent_end"] for s in samples], dtype=np.int64)
    result["cot_branch_weight"] = np.asarray([s["cot_branch_weight"] for s in samples], dtype=np.float32)

    result["latent_positions"] = _pad_2d([s["latent_positions"] for s in samples], -1, np.int64)
    result["latent_slot_mask"] = _pad_2d([s["latent_slot_mask"] for s in samples], False, np.bool_)
    result["loss_source_positions"] = _pad_2d([s["loss_source_positions"] for s in samples], -1, np.int64)
    result["loss_target_positions"] = _pad_2d([s["loss_target_positions"] for s in samples], -1, np.int64)
    result["loss_pair_mask"] = _pad_2d([s["loss_pair_mask"] for s in samples], False, np.bool_)
    result["teacher_kl_source_positions"] = _pad_2d(
        [s["teacher_kl_source_positions"] for s in samples], -1, np.int64
    )
    result["teacher_kl_target_positions"] = _pad_2d(
        [s["teacher_kl_target_positions"] for s in samples], -1, np.int64
    )
    result["teacher_kl_pair_mask"] = _pad_2d([s["teacher_kl_pair_mask"] for s in samples], False, np.bool_)
    return result


def batch_to_tuple(batch: Dict[str, np.ndarray]) -> Tuple[np.ndarray, ...]:
    return tuple(batch[name] for name in BATCH_COLUMNS)
