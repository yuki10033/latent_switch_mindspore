from __future__ import annotations

from typing import Any, Dict, Iterator, List, Sequence

import numpy as np

from latent_switch_mindspore.batch import (
    BATCH_COLUMNS,
    batch_to_tuple,
    build_effective_token_mappings,
    collate_samples,
)
from latent_switch_mindspore.records import read_records
from latent_switch_mindspore.tokens import (
    ASSISTANT_PREFIX,
    IM_END,
    LATENT_THINK_END,
    LATENT_THINK_START,
    THINK_END,
    THINK_START,
    USER_PREFIX,
    SampleSpans,
    build_spans,
    build_teacher_reference_spans,
    encode_latent_placeholders,
    get_token_constants,
    render_manual_chat,
)


def _build_structured_student_ids(row: Dict[str, Any], tokenizer: Any, token_constants: Dict[str, int]) -> List[int]:
    user_content = str(row["messages"][0]["content"])
    token_ids: List[int] = []
    token_ids.extend(tokenizer.encode(USER_PREFIX, add_special_tokens=False))
    token_ids.extend(tokenizer.encode(user_content, add_special_tokens=False))
    token_ids.extend(tokenizer.encode(IM_END, add_special_tokens=False))
    token_ids.extend(tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False))
    token_ids.append(int(token_constants["latent_start_id"]))
    token_ids.extend(
        encode_latent_placeholders(
            tokenizer,
            str(row.get("latent_pad_token", "<|endoftext|>")),
            int(row.get("n_latent_steps", 0) or 0),
        )
    )
    token_ids.append(int(token_constants["latent_end_id"]))
    token_ids.append(int(token_constants["think_start_id"]))
    token_ids.extend(tokenizer.encode(str(row["assistant_cot"]), add_special_tokens=False))
    token_ids.append(int(token_constants["think_end_id"]))
    token_ids.extend(tokenizer.encode(str(row["assistant_answer"]), add_special_tokens=False))
    token_ids.extend(tokenizer.encode(IM_END, add_special_tokens=False))
    return token_ids


def _truncate_student(token_ids: List[int], spans: SampleSpans, max_length: int) -> List[int]:
    if len(token_ids) <= max_length:
        return token_ids
    prompt_prefix = token_ids[: spans.assistant_prefix_start]
    assistant_head = token_ids[spans.assistant_prefix_start : spans.think_start + 1]
    assistant_tail = token_ids[spans.think_end :]
    mandatory = assistant_head + assistant_tail
    if len(mandatory) <= max_length:
        prefix_budget = max_length - len(mandatory)
        return prompt_prefix[-prefix_budget:] + mandatory if prefix_budget > 0 else mandatory
    if len(assistant_head) >= max_length:
        return assistant_head[:max_length]
    tail_budget = max_length - len(assistant_head)
    tail_start = max(spans.think_end, len(token_ids) - tail_budget)
    return assistant_head + token_ids[tail_start:]


def _truncate_teacher(token_ids: List[int], spans: Any, max_length: int) -> List[int]:
    if len(token_ids) <= max_length:
        return token_ids
    prompt_prefix = token_ids[: spans.assistant_prefix_start]
    assistant_head = token_ids[spans.assistant_prefix_start : spans.think_start + 1]
    assistant_tail = token_ids[spans.think_end :]
    mandatory = assistant_head + assistant_tail
    if len(mandatory) <= max_length:
        prefix_budget = max_length - len(mandatory)
        return prompt_prefix[-prefix_budget:] + mandatory if prefix_budget > 0 else mandatory
    if len(assistant_head) >= max_length:
        return assistant_head[:max_length]
    tail_budget = max_length - len(assistant_head)
    tail_start = max(spans.think_end, len(token_ids) - tail_budget)
    return assistant_head + token_ids[tail_start:]


def materialize_sample(record: Dict[str, Any], tokenizer: Any, max_length: int) -> Dict[str, Any]:
    token_constants = get_token_constants(tokenizer)
    student_ids = _build_structured_student_ids(record, tokenizer, token_constants)
    spans = build_spans(student_ids, tokenizer, token_constants)
    if len(student_ids) > max_length:
        student_ids = _truncate_student(student_ids, spans, max_length)
        spans = build_spans(student_ids, tokenizer, token_constants)
    if spans.answer_start >= spans.im_end:
        raise ValueError("Truncated sample lost the answer region")

    teacher_messages = record.get("state_align_reference_messages")
    if not isinstance(teacher_messages, list):
        teacher_messages = [
            {"role": "user", "content": str(record["messages"][0]["content"])},
            {
                "role": "assistant",
                "content": f"{THINK_START}{record['assistant_cot']}{THINK_END}{record['assistant_answer']}",
            },
        ]
    teacher_ids = tokenizer.encode(render_manual_chat(teacher_messages), add_special_tokens=False)
    teacher_spans = build_teacher_reference_spans(teacher_ids, tokenizer, token_constants)
    if len(teacher_ids) > max_length:
        teacher_ids = _truncate_teacher(teacher_ids, teacher_spans, max_length)
        teacher_spans = build_teacher_reference_spans(teacher_ids, tokenizer, token_constants)
    if teacher_spans.answer_start >= teacher_spans.im_end:
        raise ValueError("Truncated teacher reference lost the answer region")

    labels = list(student_ids)
    loss_weights = [0.0] * len(student_ids)
    prompt_mask = [False] * len(student_ids)
    latent_internal_mask = [False] * len(student_ids)
    latent_boundary_mask = [False] * len(student_ids)
    cot_mask = [False] * len(student_ids)
    answer_mask = [False] * len(student_ids)
    teacher_kl_mask = [False] * len(student_ids)

    for idx in range(spans.assistant_content_start):
        labels[idx] = -100
        prompt_mask[idx] = True

    for idx in range(spans.latent_start + 1, spans.latent_end):
        labels[idx] = -100
        latent_internal_mask[idx] = True

    latent_boundary_mask[spans.latent_start] = True
    latent_boundary_mask[spans.latent_end] = True
    loss_weights[spans.latent_start] = float(record.get("latent_start_ce_loss_weight", 1.0))
    loss_weights[spans.latent_end] = float(record.get("latent_end_ce_loss_weight", 1.0))

    cot_branch_weight = float(record.get("cot_loss_weight", 0.0))
    skip_teacher_kl = bool(record.get("skip_teacher_kl", False))
    for idx in range(spans.think_start, spans.answer_start):
        cot_mask[idx] = True
        loss_weights[idx] = cot_branch_weight
        if idx > 0 and not skip_teacher_kl:
            teacher_kl_mask[idx - 1] = True

    answer_weight = float(record.get("answer_loss_weight", 1.0))
    for idx in range(spans.answer_start, spans.im_end):
        answer_mask[idx] = True
        loss_weights[idx] = answer_weight
        if idx > 0 and not skip_teacher_kl:
            teacher_kl_mask[idx - 1] = True

    loss_weights[spans.im_end] = float(record.get("im_end_ce_loss_weight", 1.0))
    latent_positions = list(range(spans.latent_start + 1, spans.latent_end))
    valid_token_mask = [True] * len(student_ids)
    (
        loss_source_positions,
        loss_target_positions,
        teacher_kl_source_positions,
        teacher_kl_target_positions,
    ) = build_effective_token_mappings(valid_token_mask, labels, teacher_kl_mask)

    return {
        "record_id": str(record["record_id"]),
        "input_ids": [int(x) for x in student_ids],
        "labels": [int(x) for x in labels],
        "loss_weights": [float(x) for x in loss_weights],
        "attention_mask": [1] * len(student_ids),
        "position_ids": list(range(len(student_ids))),
        "prompt_mask": prompt_mask,
        "latent_internal_mask": latent_internal_mask,
        "latent_boundary_mask": latent_boundary_mask,
        "cot_mask": cot_mask,
        "answer_mask": answer_mask,
        "teacher_kl_mask": teacher_kl_mask,
        "valid_token_mask": valid_token_mask,
        "latent_pad_mask": [False] * len(student_ids),
        "teacher_target_start": int(spans.latent_end),
        "spans": spans.to_dict(),
        "teacher_spans": teacher_spans.to_dict(),
        "teacher_ids": [int(x) for x in teacher_ids],
        "latent_positions": latent_positions,
        "latent_slot_mask": [True] * len(latent_positions),
        "latent_length": len(latent_positions),
        "loss_source_positions": loss_source_positions,
        "loss_target_positions": loss_target_positions,
        "loss_pair_mask": [True] * len(loss_target_positions),
        "teacher_kl_source_positions": teacher_kl_source_positions,
        "teacher_kl_target_positions": teacher_kl_target_positions,
        "teacher_kl_pair_mask": [True] * len(teacher_kl_target_positions),
        "cot_branch_weight": cot_branch_weight,
    }


class LatentSwitchSFTSource:
    def __init__(self, records: Sequence[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    @classmethod
    def from_path(cls, path: str, tokenizer: Any, max_length: int) -> "LatentSwitchSFTSource":
        return cls(read_records(path), tokenizer=tokenizer, max_length=max_length)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return materialize_sample(self.records[index], self.tokenizer, self.max_length)


class _BatchSampler:
    def __init__(self, size: int, batch_size: int, shuffle: bool, drop_remainder: bool, seed: int = 0):
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_remainder = bool(drop_remainder)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[List[int]]:
        indices = np.arange(self.size)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        for start in range(0, self.size, self.batch_size):
            batch = indices[start : start + self.batch_size].astype(int).tolist()
            if len(batch) < self.batch_size and self.drop_remainder:
                continue
            yield batch

    def __len__(self) -> int:
        full, remainder = divmod(self.size, self.batch_size)
        return full if self.drop_remainder or remainder == 0 else full + 1


class _MindSporeSource:
    def __init__(self, source: LatentSwitchSFTSource, pad_token_id: int):
        self.source = source
        self.pad_token_id = int(pad_token_id)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.source[index]

    def collate(self, rows: Sequence[Dict[str, Any]]) -> tuple:
        return batch_to_tuple(collate_samples(rows, pad_token_id=self.pad_token_id))


def create_mindspore_dataset(
    data_path: str,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    shuffle: bool = False,
    drop_remainder: bool = False,
    num_parallel_workers: int = 1,
):
    try:
        import mindspore.dataset as ds
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError("create_mindspore_dataset requires MindSpore to be installed.") from exc

    source = LatentSwitchSFTSource.from_path(data_path, tokenizer=tokenizer, max_length=max_length)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = tokenizer.convert_tokens_to_ids(getattr(tokenizer, "pad_token", "<pad>"))
    ms_source = _MindSporeSource(source, int(pad_token_id))
    sampler = _BatchSampler(len(source), batch_size, shuffle=shuffle, drop_remainder=drop_remainder)
    return ds.GeneratorDataset(
        source=ms_source,
        column_names=list(BATCH_COLUMNS),
        num_parallel_workers=num_parallel_workers,
        batch_sampler=sampler,
        collate_fn=ms_source.collate,
        python_multiprocessing=False,
    )
