# SPDX-License-Identifier: Apache-2.0
"""Contract tests for Higgs TTS pre-tokenized rollout input.

These exercise the RL rollout path where miles sends the exact prompt token ids it
trains on. No checkpoint, tokenizer, or GPU required.
"""

import pytest

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.pretokenized import (
    build_pretokenized_state,
    is_pretokenized_prompt,
)


def test_is_pretokenized_prompt_true_for_nonempty_int_list():
    assert is_pretokenized_prompt([1, 2, 3]) is True


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"text": "hello"},
        "hello",
        [],  # empty list -> normal path
        [1, "x", 3],  # mixed -> not pre-tokenized
        None,
        ([1, 2, 3]),  # tuple is not a list
    ],
)
def test_is_pretokenized_prompt_false_for_non_int_list(inputs):
    if isinstance(inputs, tuple):
        assert is_pretokenized_prompt(inputs) is False
    else:
        assert is_pretokenized_prompt(inputs) is False


def test_build_pretokenized_state_uses_ids_verbatim():
    state = build_pretokenized_state(
        [5, 6, 7],
        {
            "max_new_tokens": 64,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "seed": 1,
        },
    )
    assert isinstance(state, HiggsTtsState)
    assert state.prompt_token_ids == [5, 6, 7]
    # pre-tokenized path carries no reference-audio / text-tokenization artifacts
    assert state.reference_codes_delayed is None
    assert state.reference_waveform is None
    assert state.target_text is None
    assert state.reference_text is None
    # generation params flow through
    assert state.max_new_tokens == 64
    assert state.temperature == 0.8
    assert state.top_p == 0.9
    assert state.top_k == 50
    assert state.seed == 1


def test_build_pretokenized_state_param_defaults():
    state = build_pretokenized_state([1], {})
    assert state.max_new_tokens == 2048
    assert state.temperature == 1.0
    assert state.top_p is None
    assert state.top_k is None
    assert state.seed is None
    assert state.num_codebooks == 8
    assert state.codebook_size == 1026


def test_build_pretokenized_state_custom_codebooks():
    state = build_pretokenized_state([9], None, num_codebooks=4, codebook_size=512)
    assert state.num_codebooks == 4
    assert state.codebook_size == 512
    assert state.prompt_token_ids == [9]


def test_build_pretokenized_state_to_dict_roundtrips_prompt():
    state = build_pretokenized_state([10, 11], {"max_new_tokens": 32})
    data = state.to_dict()
    assert data["prompt_token_ids"] == [10, 11]
    # no reference artifacts should be serialized for the pre-tokenized path
    assert "reference_codes_delayed" not in data
    assert "reference_waveform" not in data
    restored = HiggsTtsState.from_dict(data)
    assert restored.prompt_token_ids == [10, 11]
    assert restored.max_new_tokens == 32
