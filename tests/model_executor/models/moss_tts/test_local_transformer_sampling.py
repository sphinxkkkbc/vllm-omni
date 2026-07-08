from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import (
    MossTTSLocalDepthTransformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEMPERATURE = 1.7
TOP_K = 25
TOP_P = 0.8


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _noise(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float32).exponential_(generator=_generator(seed))


def _make_depth_transformer(hidden_size: int = 8) -> MossTTSLocalDepthTransformer:
    cfg = SimpleNamespace(
        n_embd=hidden_size,
        n_head=2,
        n_inner=hidden_size * 2,
        layer_norm_epsilon=1e-5,
        rope_base=10000.0,
    )
    torch.manual_seed(0)
    return MossTTSLocalDepthTransformer(cfg, hidden_size=hidden_size).eval()


def _make_heads(
    *,
    hidden_size: int,
    n_vq: int,
    audio_vocab_size: int,
) -> tuple[nn.ModuleList, nn.ModuleList, nn.Linear]:
    torch.manual_seed(1)
    audio_lm_heads = nn.ModuleList([nn.Linear(hidden_size, audio_vocab_size) for _ in range(n_vq)]).eval()
    audio_embeddings = nn.ModuleList([nn.Embedding(audio_vocab_size, hidden_size) for _ in range(n_vq)]).eval()
    local_text_lm_head = nn.Linear(hidden_size, 2).eval()
    return audio_lm_heads, audio_embeddings, local_text_lm_head


def test_generate_frame_uses_sampling_noise_first(monkeypatch) -> None:
    hidden_size = 8
    n_vq = 2
    audio_vocab_size = 5
    batch_size = 1
    transformer = _make_depth_transformer(hidden_size)
    audio_lm_heads, audio_embeddings, local_text_lm_head = _make_heads(
        hidden_size=hidden_size,
        n_vq=n_vq,
        audio_vocab_size=audio_vocab_size,
    )
    torch.manual_seed(7)
    backbone_last_hidden = torch.randn(batch_size, hidden_size)
    base_kwargs = dict(
        n_vq=n_vq,
        do_sample=True,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    noise_shape = (batch_size, 1 + n_vq, audio_vocab_size)

    def fail_multinomial(*args, **kwargs):
        raise AssertionError("sampling_noise path must not call torch.multinomial")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)
    sampling_noise = _noise(noise_shape, seed=7)
    should_continue_a, codes_a = transformer.generate_frame(
        backbone_last_hidden,
        audio_lm_heads,
        audio_embeddings,
        local_text_lm_head,
        sampling_noise=sampling_noise,
        **base_kwargs,
    )
    should_continue_b, codes_b = transformer.generate_frame(
        backbone_last_hidden,
        audio_lm_heads,
        audio_embeddings,
        local_text_lm_head,
        sampling_noise=_noise(noise_shape, seed=7),
        **base_kwargs,
    )
    should_continue_c, codes_c = transformer.generate_frame(
        backbone_last_hidden,
        audio_lm_heads,
        audio_embeddings,
        local_text_lm_head,
        sampling_noise=_noise(noise_shape, seed=7),
        generator=_generator(42),
        **base_kwargs,
    )

    assert torch.equal(should_continue_a, should_continue_b)
    assert torch.equal(should_continue_a, should_continue_c)
    assert torch.equal(codes_a, codes_b)
    assert torch.equal(codes_a, codes_c)
    assert codes_a.shape == (batch_size, n_vq)


def test_generate_frame_eager_generator_paths(monkeypatch) -> None:
    hidden_size = 8
    n_vq = 3
    audio_vocab_size = 6
    batch_size = 2

    transformer = _make_depth_transformer(hidden_size)
    audio_lm_heads, audio_embeddings, local_text_lm_head = _make_heads(
        hidden_size=hidden_size,
        n_vq=n_vq,
        audio_vocab_size=audio_vocab_size,
    )
    torch.manual_seed(7)
    backbone_last_hidden = torch.randn(batch_size, hidden_size)
    base_kwargs = dict(
        n_vq=n_vq,
        do_sample=True,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    captured_generators = []

    def fake_multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
        captured_generators.append(generator)
        return torch.zeros((input.shape[0], num_samples), dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    generator = _generator(42)
    should_continue, codes = transformer.generate_frame(
        backbone_last_hidden,
        audio_lm_heads,
        audio_embeddings,
        local_text_lm_head,
        generator=generator,
        **base_kwargs,
    )
    assert captured_generators
    assert all(captured_generator is generator for captured_generator in captured_generators)
    assert should_continue.shape == (batch_size,)
    assert codes.shape == (batch_size, n_vq)


def test_generate_frame_accepts_tensor_history_codes() -> None:
    hidden_size = 8
    n_vq = 3
    audio_vocab_size = 6
    batch_size = 2

    transformer = _make_depth_transformer(hidden_size)
    audio_lm_heads, audio_embeddings, local_text_lm_head = _make_heads(
        hidden_size=hidden_size,
        n_vq=n_vq,
        audio_vocab_size=audio_vocab_size,
    )
    torch.manual_seed(5)
    backbone_last_hidden = torch.randn(batch_size, hidden_size)
    history_codes = torch.tensor(
        [
            [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [0, 0, 1, 1, 2]],
            [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [2, 2, 3, 3, 4]],
        ],
        dtype=torch.long,
    )

    should_continue, codes = transformer.generate_frame(
        backbone_last_hidden,
        audio_lm_heads,
        audio_embeddings,
        local_text_lm_head,
        n_vq=n_vq,
        do_sample=False,
        repetition_penalty=1.2,
        history_codes=history_codes,
    )

    assert should_continue.shape == (batch_size,)
    assert codes.shape == (batch_size, n_vq)
