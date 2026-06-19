# SPDX-License-Identifier: Apache-2.0
"""Pre-tokenized prompt handling for Higgs TTS rollout.

Miles RL rollout sends the exact prompt token ids it trains on, so those ids must
bypass the text tokenizer and reference-audio assembly to keep rollout and training
tokens identical. This mirrors the Qwen3-Omni thinker's pre-tokenized path.

Kept dependency-light (only :class:`HiggsTtsState`) so the contract is unit-testable
without a checkpoint, tokenizer, or GPU.
"""

from __future__ import annotations

from typing import Any

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState


def is_pretokenized_prompt(inputs: Any) -> bool:
    """True when a rollout request carries pre-tokenized prompt ids.

    A non-empty list of ints is treated as pre-tokenized; message dicts, raw strings,
    and empty inputs go through the normal text-tokenization path.
    """
    return (
        isinstance(inputs, list)
        and bool(inputs)
        and all(isinstance(token, int) for token in inputs)
    )


def build_pretokenized_state(
    token_ids: list[int],
    params: dict[str, Any] | None,
    *,
    num_codebooks: int = 8,
    codebook_size: int = 1026,
) -> HiggsTtsState:
    """Build Higgs TTS state directly from pre-tokenized prompt ids.

    Equivalent to the no-reference text branch of the preprocessing stage, but skips
    text tokenization and reference-audio handling so the TTS actor runs the exact
    prompt tokens the RL trainer computes gradients on.
    """
    params = params or {}
    return HiggsTtsState(
        prompt_token_ids=list(token_ids),
        reference_codes_delayed=None,
        reference_waveform=None,
        reference_code_cache_key=None,
        target_text=None,
        reference_text=None,
        uploaded_voice_name=None,
        uploaded_voice_created_at=None,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_new_tokens=int(params.get("max_new_tokens", 2048)),
        temperature=float(params.get("temperature", 1.0)),
        top_p=params.get("top_p"),
        top_k=params.get("top_k"),
        seed=params.get("seed"),
    )


__all__ = ["is_pretokenized_prompt", "build_pretokenized_state"]
