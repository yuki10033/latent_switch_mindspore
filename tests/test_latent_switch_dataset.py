from __future__ import annotations

import json

import numpy as np
import pytest

from latent_switch_mindspore.batch import collate_samples
from latent_switch_mindspore.dataset import materialize_sample
from latent_switch_mindspore.records import SFTBuildConfig, build_sft_record, write_jsonl
from latent_switch_mindspore.tokens import SPECIAL_TOKENS, build_spans, validate_tokenizer_contract


class FakeTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.additional_special_tokens = []
        self.pad_token = "<pad>"
        self.unk_token_id = -1
        for token in ["<pad>", *SPECIAL_TOKENS]:
            self._add(token)
        self.pad_token_id = self.vocab["<pad>"]

    def _add(self, token):
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inv_vocab[idx] = token
        return self.vocab[token]

    def add_special_tokens(self, payload):
        count = 0
        for token in payload.get("additional_special_tokens", []):
            if token not in self.vocab:
                count += 1
            self._add(str(token))
            if str(token) not in self.additional_special_tokens:
                self.additional_special_tokens.append(str(token))
        if "pad_token" in payload:
            self.pad_token = payload["pad_token"]
            self.pad_token_id = self._add(self.pad_token)
        return count

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(str(token), self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        specials = sorted(self.vocab, key=len, reverse=True)
        while i < len(text):
            matched = None
            for token in specials:
                if text.startswith(token, i):
                    matched = token
                    break
            if matched is not None:
                ids.append(self.vocab[matched])
                i += len(matched)
                continue
            if text[i].isspace():
                i += 1
                continue
            j = i + 1
            while j < len(text) and not text[j].isspace() and not any(text.startswith(tok, j) for tok in specials):
                j += 1
            ids.append(self._add(text[i:j]))
            i = j
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inv_vocab[int(i)] for i in ids)


def distilled_row():
    return {
        "uid": "row-1",
        "question": "What is two plus two?",
        "stage1": {"correct_insight": "add the two numbers carefully"},
        "stage2": {
            "distilled_cot": "2 + 2 = 4.",
            "answer": "The answer is 4.",
            "validation": {"is_correct": True},
        },
        "token_stats": {"compression_ratio_vs_primary_output": 0.5, "distilled_cot_tokens": 4},
    }


def test_special_tokens_encode_once():
    tokenizer = FakeTokenizer()
    constants = validate_tokenizer_contract(tokenizer)
    for key in ["latent_start_id", "latent_end_id", "think_start_id", "think_end_id", "im_end_id"]:
        assert isinstance(constants[key], int)


def test_build_sft_record_computes_latent_steps_and_filters_missing():
    tokenizer = FakeTokenizer()
    record, reason = build_sft_record(distilled_row(), tokenizer, SFTBuildConfig(latent_min=1, latent_max=128))
    assert reason == "ok"
    assert record is not None
    assert record["n_latent_steps"] == 2
    missing, reason = build_sft_record({"question": "x"}, tokenizer)
    assert missing is None
    assert reason == "missing_cot"


def test_materialize_masks_follow_chapter_invariants():
    tokenizer = FakeTokenizer()
    record, _ = build_sft_record(distilled_row(), tokenizer, SFTBuildConfig(latent_min=1, latent_max=128))
    sample = materialize_sample(record, tokenizer, max_length=256)
    spans = sample["spans"]

    for idx, is_prompt in enumerate(sample["prompt_mask"]):
        if is_prompt:
            assert sample["labels"][idx] == -100
    for pos in sample["latent_positions"]:
        assert sample["latent_internal_mask"][pos]
        assert sample["labels"][pos] == -100
        assert not sample["teacher_kl_mask"][pos]
    assert sample["latent_boundary_mask"][spans["latent_start"]]
    assert sample["latent_boundary_mask"][spans["latent_end"]]
    assert sample["labels"][spans["latent_start"]] != -100
    assert sample["labels"][spans["latent_end"]] != -100
    assert not sample["answer_mask"][spans["im_end"]]
    assert np.asarray(sample["attention_mask"]).astype(bool).tolist() == sample["valid_token_mask"]


def test_span_validation_rejects_duplicate_boundary():
    tokenizer = FakeTokenizer()
    record, _ = build_sft_record(distilled_row(), tokenizer)
    sample = materialize_sample(record, tokenizer, max_length=256)
    token_ids = list(sample["input_ids"])
    token_ids.insert(sample["spans"]["latent_start"], tokenizer.convert_tokens_to_ids("<latent_think>"))
    with pytest.raises(ValueError, match="Expected exactly one"):
        build_spans(token_ids, tokenizer, validate_tokenizer_contract(tokenizer))


def test_batch_padding_shapes_and_masks():
    tokenizer = FakeTokenizer()
    record1, _ = build_sft_record(distilled_row(), tokenizer)
    row2 = distilled_row()
    row2["uid"] = "row-2"
    row2["question"] = "Compute three plus five with a short check."
    row2["stage1"]["correct_insight"] = "add three and five then verify"
    record2, _ = build_sft_record(row2, tokenizer)
    samples = [materialize_sample(record1, tokenizer, 256), materialize_sample(record2, tokenizer, 256)]
    batch = collate_samples(samples, pad_token_id=tokenizer.pad_token_id)
    assert batch["input_ids"].ndim == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert np.array_equal(batch["attention_mask"].astype(bool), batch["valid_token_mask"])
    assert batch["latent_positions"].shape[0] == 2
    assert batch["loss_source_positions"].shape == batch["loss_target_positions"].shape


def test_jsonl_round_trip_for_integration(tmp_path):
    tokenizer = FakeTokenizer()
    records = []
    for idx in range(2):
        row = distilled_row()
        row["uid"] = f"row-{idx}"
        record, reason = build_sft_record(row, tokenizer)
        assert reason == "ok"
        records.append(record)
    path = tmp_path / "sft_train.jsonl"
    assert write_jsonl(path, records) == 2
    loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    samples = [materialize_sample(row, tokenizer, max_length=256) for row in loaded]
    batch = collate_samples(samples, pad_token_id=tokenizer.pad_token_id)
    assert batch["input_ids"].shape[0] == 2
    assert batch["latent_slot_mask"].any()
