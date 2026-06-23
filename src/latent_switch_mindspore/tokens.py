from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence


LATENT_THINK_START = "<latent_think>"
LATENT_THINK_END = "</latent_think>"
THINK_START = "<think>"
THINK_END = "</think>"
IM_START = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
DEFAULT_LATENT_PAD_TOKEN = "<|endoftext|>"

USER_PREFIX = f"{IM_START}user\n"
ASSISTANT_PREFIX = f"{IM_START}assistant\n"
IM_END = f"{IM_END_TOKEN}\n"

SPECIAL_TOKENS = (
    LATENT_THINK_START,
    LATENT_THINK_END,
    THINK_START,
    THINK_END,
    IM_START,
    IM_END_TOKEN,
    DEFAULT_LATENT_PAD_TOKEN,
)


@dataclass(frozen=True)
class SampleSpans:
    assistant_prefix_start: int
    assistant_content_start: int
    latent_start: int
    latent_end: int
    think_start: int
    think_end: int
    answer_start: int
    im_end: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TeacherReferenceSpans:
    assistant_prefix_start: int
    assistant_content_start: int
    think_start: int
    think_end: int
    answer_start: int
    im_end: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def render_manual_chat(messages: Sequence[Dict[str, str]]) -> str:
    parts: List[str] = []
    for message in messages:
        role = str(message["role"])
        content = str(message["content"])
        parts.append(f"{IM_START}{role}\n{content}{IM_END}")
    return "".join(parts)


def render_student_messages(user_content: str, assistant_content: str) -> str:
    return f"{USER_PREFIX}{user_content}{IM_END}{ASSISTANT_PREFIX}{assistant_content}{IM_END}"


def render_prompt_only(user_content: str) -> str:
    return f"{USER_PREFIX}{user_content}{IM_END}{ASSISTANT_PREFIX}"


def find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    if not subsequence:
        return -1
    limit = len(sequence) - len(subsequence) + 1
    target = list(subsequence)
    for idx in range(max(limit, 0)):
        if list(sequence[idx : idx + len(target)]) == target:
            return idx
    return -1


def _token_to_id(tokenizer: Any, token: str) -> Optional[int]:
    if tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk_id:
        return None
    try:
        token_id = int(token_id)
    except Exception:
        return None
    if token_id < 0:
        return None
    return token_id


def ensure_special_tokens(tokenizer: Any, pad_token: str = "<pad>") -> int:
    """Register required structural tokens when the tokenizer supports it."""
    if tokenizer is None:
        raise ValueError("tokenizer is required")

    existing = [str(token) for token in getattr(tokenizer, "additional_special_tokens", []) or []]
    to_add = [token for token in SPECIAL_TOKENS if _token_to_id(tokenizer, token) is None and token not in existing]
    special_tokens: Dict[str, Any] = {}
    if to_add:
        special_tokens["additional_special_tokens"] = existing + to_add
    if getattr(tokenizer, "pad_token", None) is None:
        special_tokens["pad_token"] = pad_token

    if special_tokens and hasattr(tokenizer, "add_special_tokens"):
        return int(tokenizer.add_special_tokens(special_tokens))
    return 0


def validate_tokenizer_contract(tokenizer: Any) -> Dict[str, int]:
    ensure_special_tokens(tokenizer)
    names = {
        "latent_start_id": LATENT_THINK_START,
        "latent_end_id": LATENT_THINK_END,
        "think_start_id": THINK_START,
        "think_end_id": THINK_END,
        "im_start_id": IM_START,
        "im_end_id": IM_END_TOKEN,
        "latent_pad_id": DEFAULT_LATENT_PAD_TOKEN,
    }
    constants: Dict[str, int] = {}
    for name, token in names.items():
        token_id = _token_to_id(tokenizer, token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required special token: {token}")
        encoded = list(tokenizer.encode(token, add_special_tokens=False))
        if encoded != [token_id]:
            raise ValueError(f"Tokenizer must encode {token!r} as one token. Got {encoded}.")
        constants[name] = int(token_id)

    adjacency = f"{LATENT_THINK_START}{DEFAULT_LATENT_PAD_TOKEN}{LATENT_THINK_END}{THINK_START}"
    adjacency_ids = [int(x) for x in tokenizer.encode(adjacency, add_special_tokens=False)]
    if adjacency_ids.count(constants["latent_start_id"]) != 1:
        raise ValueError("Tokenizer split or duplicated <latent_think> in adjacent context.")
    if adjacency_ids.count(constants["latent_end_id"]) != 1:
        raise ValueError("Tokenizer split or duplicated </latent_think> in adjacent context.")
    if adjacency_ids.count(constants["think_start_id"]) != 1:
        raise ValueError("Tokenizer split or duplicated <think> in adjacent context.")
    return constants


def get_token_constants(tokenizer: Any) -> Dict[str, int]:
    return validate_tokenizer_contract(tokenizer)


def encode_latent_placeholders(tokenizer: Any, latent_pad_token: str, n_latent_steps: int) -> List[int]:
    if n_latent_steps <= 0:
        return []
    token_id = _token_to_id(tokenizer, latent_pad_token)
    if token_id is not None:
        return [int(token_id)] * int(n_latent_steps)
    return list(tokenizer.encode(latent_pad_token * int(n_latent_steps), add_special_tokens=False))


def build_spans(token_ids: Sequence[int], tokenizer: Any, token_constants: Dict[str, int]) -> SampleSpans:
    ids = list(token_ids)
    latent_start_id = int(token_constants["latent_start_id"])
    latent_end_id = int(token_constants["latent_end_id"])
    think_start_id = int(token_constants["think_start_id"])
    think_end_id = int(token_constants["think_end_id"])
    im_end_id = int(token_constants["im_end_id"])

    for name, token_id in (
        ("<latent_think>", latent_start_id),
        ("</latent_think>", latent_end_id),
        ("<think>", think_start_id),
        ("</think>", think_end_id),
    ):
        if ids.count(token_id) != 1:
            raise ValueError(f"Expected exactly one {name} token in tokenized sample")

    assistant_prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
    assistant_prefix_start = find_subsequence(ids, assistant_prefix_ids)
    if assistant_prefix_start < 0:
        raise ValueError("Assistant prefix not found in tokenized sample")

    assistant_content_start = assistant_prefix_start + len(assistant_prefix_ids)
    latent_start = ids.index(latent_start_id)
    latent_end = ids.index(latent_end_id)
    think_start = ids.index(think_start_id)
    think_end = ids.index(think_end_id)
    if not (assistant_content_start <= latent_start < latent_end < think_start < think_end):
        raise ValueError(
            "Invalid assistant boundary order: "
            f"assistant_content_start={assistant_content_start}, latent_start={latent_start}, "
            f"latent_end={latent_end}, think_start={think_start}, think_end={think_end}"
        )
    if im_end_id not in ids:
        raise ValueError("Missing <|im_end|> token in tokenized sample")
    im_end = len(ids) - 1 - list(reversed(ids)).index(im_end_id)
    answer_start = think_end + 1
    return SampleSpans(
        assistant_prefix_start=assistant_prefix_start,
        assistant_content_start=assistant_content_start,
        latent_start=latent_start,
        latent_end=latent_end,
        think_start=think_start,
        think_end=think_end,
        answer_start=answer_start,
        im_end=im_end,
    )


def build_teacher_reference_spans(
    token_ids: Sequence[int],
    tokenizer: Any,
    token_constants: Dict[str, int],
) -> TeacherReferenceSpans:
    ids = list(token_ids)
    think_start_id = int(token_constants["think_start_id"])
    think_end_id = int(token_constants["think_end_id"])
    im_end_id = int(token_constants["im_end_id"])

    if ids.count(think_start_id) != 1:
        raise ValueError("Expected exactly one <think> token in teacher reference")
    if ids.count(think_end_id) != 1:
        raise ValueError("Expected exactly one </think> token in teacher reference")
    assistant_prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
    assistant_prefix_start = find_subsequence(ids, assistant_prefix_ids)
    if assistant_prefix_start < 0:
        raise ValueError("Assistant prefix not found in teacher reference")

    assistant_content_start = assistant_prefix_start + len(assistant_prefix_ids)
    think_start = ids.index(think_start_id)
    think_end = ids.index(think_end_id)
    if not (assistant_content_start <= think_start < think_end):
        raise ValueError(
            "Invalid teacher boundary order: "
            f"assistant_content_start={assistant_content_start}, think_start={think_start}, think_end={think_end}"
        )
    if im_end_id not in ids:
        raise ValueError("Missing <|im_end|> token in teacher reference")
    im_end = len(ids) - 1 - list(reversed(ids)).index(im_end_id)
    return TeacherReferenceSpans(
        assistant_prefix_start=assistant_prefix_start,
        assistant_content_start=assistant_content_start,
        think_start=think_start,
        think_end=think_end,
        answer_start=think_end + 1,
        im_end=im_end,
    )
