"""Unit tests for the KV-cache (kamui.model.kv_cache) and cached decoding.

Coverage target:
    kamui/model/kv_cache.py — 100%

The central property: decoding with a cache gives the same logits as a full
forward pass, under every positional-encoding scheme, and cached generation
emits exactly the same tokens as uncached generation.
"""

from __future__ import annotations

import pytest
import torch

from kamui.evaluate.generation import generate
from kamui.hooks.manager import HookManager
from kamui.mechinterp.steering import FeatureSteerer
from kamui.model.config import ModelConfig
from kamui.model.embedding import (
    Embedding,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
)
from kamui.model.kv_cache import KVCache, LayerKVCache
from kamui.model.transformer import KAMUITransformer

_ENCODINGS = ["learned", "sinusoidal", "rope"]


def _model(positional_encoding: str = "learned", **overrides: object) -> KAMUITransformer:
    torch.manual_seed(0)
    base = dict(
        n_layers=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        vocab_size=40,
        context_length=8,
        positional_encoding=positional_encoding,
        dropout=0.0,
    )
    base.update(overrides)
    return KAMUITransformer(ModelConfig(**base)).eval()  # type: ignore[arg-type]


class _Tok:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 40 for c in text]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


# ===========================================================================
# Cache containers
# ===========================================================================


class TestCacheContainers:
    def test_layer_cache_appends_along_sequence(self) -> None:
        layer = LayerKVCache()
        assert layer.length == 0
        k1, v1 = torch.ones(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)
        keys, values = layer.append(k1, v1)
        assert keys.shape == (1, 2, 3, 4) and layer.length == 3
        keys, values = layer.append(torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4))
        assert keys.shape == (1, 2, 4, 4) and values.shape == (1, 2, 4, 4)
        assert layer.length == 4

    def test_layer_cache_reset(self) -> None:
        layer = LayerKVCache()
        layer.append(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2))
        layer.reset()
        assert layer.length == 0 and layer.keys is None and layer.values is None

    def test_kv_cache_length_and_reset(self) -> None:
        cache = KVCache(3)
        assert len(cache.layers) == 3 and cache.length == 0
        for layer in cache.layers:
            layer.append(torch.ones(1, 1, 5, 2), torch.ones(1, 1, 5, 2))
        assert cache.length == 5
        cache.reset()
        assert cache.length == 0

    def test_kv_cache_rejects_non_positive_layers(self) -> None:
        with pytest.raises(ValueError, match="n_layers must be > 0"):
            KVCache(0)

    def test_model_new_cache_matches_depth(self) -> None:
        model = _model(n_layers=3)
        assert len(model.new_cache().layers) == 3


# ===========================================================================
# Position offsets
# ===========================================================================


class TestPositionOffsets:
    def test_learned_offset_slices_table(self) -> None:
        pe = LearnedPositionalEncoding(context_length=8, d_model=4)
        assert torch.equal(pe(2, offset=3), pe.weight[3:5])

    def test_sinusoidal_offset_slices_table(self) -> None:
        pe = SinusoidalPositionalEncoding(context_length=8, d_model=4)
        assert torch.equal(pe(2, offset=3), pe.pe[3:5])

    def test_rope_offset_matches_full_rotation(self) -> None:
        rope = RotaryPositionalEncoding(d_head=4, context_length=8)
        x = torch.randn(1, 2, 8, 4)
        full = rope(x)
        assert torch.allclose(rope(x[:, :, 5:7], offset=5), full[:, :, 5:7], atol=1e-6)

    @pytest.mark.parametrize("cls", [LearnedPositionalEncoding, SinusoidalPositionalEncoding])
    def test_table_offset_errors(self, cls: type) -> None:
        pe = cls(context_length=8, d_model=4)
        with pytest.raises(ValueError, match="offset must be >= 0"):
            pe(2, offset=-1)
        with pytest.raises(ValueError, match="exceeds context_length"):
            pe(2, offset=7)

    def test_rope_offset_errors(self) -> None:
        rope = RotaryPositionalEncoding(d_head=4, context_length=8)
        with pytest.raises(ValueError, match="offset must be >= 0"):
            rope(torch.randn(1, 1, 2, 4), offset=-1)
        with pytest.raises(ValueError, match="exceeds context_length"):
            rope(torch.randn(1, 1, 2, 4), offset=7)

    def test_embedding_offset_errors(self) -> None:
        embed = Embedding(_model().config)
        with pytest.raises(ValueError, match="offset must be >= 0"):
            embed(torch.zeros(1, 2, dtype=torch.long), offset=-1)
        with pytest.raises(ValueError, match="exceeds context_length"):
            embed(torch.zeros(1, 2, dtype=torch.long), offset=7)


# ===========================================================================
# Cached forward == full forward
# ===========================================================================


class TestCachedForwardEquivalence:
    @pytest.mark.parametrize("encoding", _ENCODINGS)
    def test_token_by_token_matches_full_forward(self, encoding: str) -> None:
        model = _model(encoding)
        ids = torch.randint(0, 40, (1, 8))
        with torch.no_grad():
            full = model(ids)  # (1, 8, V)
            cache = model.new_cache()
            steps = [model(ids[:, :3], cache=cache)]  # prefill 3 positions
            steps += [model(ids[:, t : t + 1], cache=cache) for t in range(3, 8)]
        cached = torch.cat(steps, dim=1)
        assert cache.length == 8
        assert torch.allclose(cached, full, atol=1e-5)

    @pytest.mark.parametrize("encoding", _ENCODINGS)
    def test_multi_token_chunks_match_full_forward(self, encoding: str) -> None:
        # Chunks of 2, 3, 1, 2 exercise the mask slice with S > 1 and an offset.
        model = _model(encoding)
        ids = torch.randint(0, 40, (2, 8))
        with torch.no_grad():
            full = model(ids)
            cache = model.new_cache()
            bounds = [(0, 2), (2, 5), (5, 6), (6, 8)]
            cached = torch.cat([model(ids[:, a:b], cache=cache) for a, b in bounds], dim=1)
        assert torch.allclose(cached, full, atol=1e-5)

    def test_rmsnorm_model_also_matches(self) -> None:
        model = _model("rope", normalization="rmsnorm")
        ids = torch.randint(0, 40, (1, 6))
        with torch.no_grad():
            full = model(ids)
            cache = model.new_cache()
            cached = torch.cat([model(ids[:, t : t + 1], cache=cache) for t in range(6)], dim=1)
        assert torch.allclose(cached, full, atol=1e-5)

    def test_attention_weights_span_cached_keys(self) -> None:
        model = _model()
        layer = LayerKVCache()
        attn = model.blocks[0].attn
        with torch.no_grad():
            attn(torch.randn(1, 3, 16), cache=layer)
            _, weights = attn(torch.randn(1, 1, 16), return_weights=True, cache=layer)
        assert weights.shape == (1, 4, 1, 4)  # 1 new query over 3 cached + 1 new keys

    def test_wrong_layer_count_raises(self) -> None:
        model = _model()
        with pytest.raises(ValueError, match="cache has 3 layers"):
            model(torch.zeros(1, 2, dtype=torch.long), cache=KVCache(3))

    def test_overflowing_the_context_raises(self) -> None:
        model = _model()
        cache = model.new_cache()
        with torch.no_grad():
            model(torch.zeros(1, 8, dtype=torch.long), cache=cache)
            with pytest.raises(ValueError, match="exceeds context_length"):
                model(torch.zeros(1, 1, dtype=torch.long), cache=cache)

    def test_weights_hook_forwards_the_cache(self) -> None:
        # Capturing attention weights must not silently bypass the cache.
        model = _model()
        cache = model.new_cache()
        with torch.no_grad(), HookManager(model) as hooks:
            hooks.attach("blocks.0.attn", "weights")
            model(torch.zeros(1, 3, dtype=torch.long), cache=cache)
            model(torch.zeros(1, 1, dtype=torch.long), cache=cache)
            weights = hooks.get("blocks.0.attn.weights")
        assert cache.length == 4
        assert weights.shape == (1, 4, 1, 4)


# ===========================================================================
# Cached generation == uncached generation
# ===========================================================================


class TestCachedGeneration:
    @pytest.mark.parametrize("encoding", _ENCODINGS)
    def test_greedy_matches_uncached_past_the_context(self, encoding: str) -> None:
        # 3 prompt tokens + 12 new tokens runs well past context_length=8,
        # exercising the cache restart that mirrors uncached window cropping.
        model = _model(encoding)
        tok = _Tok()
        plain = generate(model, tok, "abc", max_new_tokens=12)
        cached = generate(model, tok, "abc", max_new_tokens=12, use_cache=True)
        assert cached == plain

    def test_long_prompt_matches_uncached(self) -> None:
        model = _model()
        tok = _Tok()
        prompt = "abcdefghijk"  # 11 tokens > context_length
        plain = generate(model, tok, prompt, max_new_tokens=4)
        assert generate(model, tok, prompt, max_new_tokens=4, use_cache=True) == plain

    def test_decoding_controls_match_uncached(self) -> None:
        model = _model("rope")
        tok = _Tok()
        kwargs = dict(max_new_tokens=10, repetition_penalty=1.5, stop_token_ids=[7])
        plain = generate(model, tok, "hello", **kwargs)  # type: ignore[arg-type]
        cached = generate(model, tok, "hello", use_cache=True, **kwargs)  # type: ignore[arg-type]
        assert cached == plain

    def test_steered_generation_matches_uncached(self) -> None:
        model = _model()
        steerer = FeatureSteerer(model)
        direction = torch.randn(16)
        args = (_Tok(), "abc", "blocks.0.ffn.output", direction)
        plain = steerer.generate_steered(*args, coefficient=3.0, max_new_tokens=6)
        cached = steerer.generate_steered(*args, coefficient=3.0, max_new_tokens=6, use_cache=True)
        assert cached == plain
