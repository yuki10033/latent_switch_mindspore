"""MindSpore-ready dataset utilities for latent-switch SFT records."""

from latent_switch_mindspore.dataset import (
    LatentSwitchSFTSource,
    create_mindspore_dataset,
    materialize_sample,
)
from latent_switch_mindspore.records import SFTBuildConfig, build_sft_record

__all__ = [
    "LatentSwitchSFTSource",
    "SFTBuildConfig",
    "build_sft_record",
    "create_mindspore_dataset",
    "materialize_sample",
]
