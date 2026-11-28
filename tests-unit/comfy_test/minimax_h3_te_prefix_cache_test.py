import uuid
from dataclasses import dataclass

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.text_encoders.llama as llama  # noqa: E402
import comfy.text_encoders.minimax as minimax  # noqa: E402
import comfy.text_encoders.qwen3vl as qwen3vl  # noqa: E402


@dataclass
class _Tiny32(llama.Qwen3VL_32BConfig):
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 32
    rope_dims = [8, 4, 4]


@dataclass
class _Tiny4(llama.Qwen3VL_4BConfig):
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 32
    rope_dims = [8, 4, 4]


VISION = dict(hidden_size=128, intermediate_size=128, depth=3, deepstack_visual_indexes=[0, 1, 2])


def rnd(seed, *shape):
    # same content, a new tensor object on every call
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed))


SCENES = {
    "img1": lambda: dict(images=[rnd(1, 1, 64, 64, 3)]),
    "img2": lambda: dict(images=[rnd(1, 1, 64, 64, 3), rnd(2, 1, 64, 96, 3)]),
    "img2_swapped": lambda: dict(images=[rnd(2, 1, 64, 96, 3), rnd(1, 1, 64, 64, 3)]),
    "img2_edited": lambda: dict(images=[rnd(1, 1, 64, 64, 3), rnd(5, 1, 64, 96, 3)]),
    "ref2va": lambda: dict(minimax_ref_items=[{"type": "image", "data": rnd(1, 1, 64, 64, 3)}, {"type": "audio"},
                                              {"type": "video", "data": rnd(3, 4, 64, 96, 3)}]),
}


def build(monkeypatch, kind):
    monkeypatch.setitem(qwen3vl.QWEN3VL_CONFIGS, "qwen3vl_32b", _Tiny32)
    monkeypatch.setitem(qwen3vl.QWEN3VL_CONFIGS, "qwen3vl_4b", _Tiny4)
    monkeypatch.setitem(qwen3vl.QWEN3VL_VISION, "qwen3vl_32b", VISION)
    monkeypatch.setitem(qwen3vl.QWEN3VL_VISION, "qwen3vl_4b", VISION)
    monkeypatch.setattr(minimax, "PH3_IMAGE_HIDDEN_STATES", [1, 2, 3, 4, 6])
    monkeypatch.setattr(minimax, "PH3_IMAGE_TEXT_DIM", 48)
    monkeypatch.setattr(minimax, "PREFIX_CACHE", True)
    torch.manual_seed(0)
    if kind == "32b":
        te, tok = minimax.te()(device="cpu", dtype=torch.float32), minimax.MiniMaxH3Tokenizer()
    else:
        te, tok = minimax.te(te_model=minimax.PH3ImageTEModel)(device="cpu", dtype=torch.float32), minimax.PH3ImageTokenizer()
    with torch.no_grad():
        for name, p in te.named_parameters():
            p.normal_(0, 0.02)
            if "norm" in name and name.endswith("weight"):
                p.fill_(1.0)
            if name.endswith("_std"):
                p.fill_(1.0)
    clip = getattr(te, te.clip)
    calls = {"visual": 0, "llm_tokens": 0}

    visual = clip.transformer.visual.forward
    def count_visual(*a, **k):
        calls["visual"] += 1
        return visual(*a, **k)
    clip.transformer.visual.forward = count_visual

    model = clip.transformer.model.forward
    def count_llm(*a, **k):
        calls["llm_tokens"] += k["embeds"].shape[1]
        return model(*a, **k)
    clip.transformer.model.forward = count_llm
    return te, clip, tok, calls


@pytest.fixture(params=["32b", "ph3"])
def tiny(request, monkeypatch):
    return build(monkeypatch, request.param)


@pytest.fixture
def tiny32(monkeypatch):
    return build(monkeypatch, "32b")


def encode(te, tok, text, scene):
    t = tok.tokenize_with_weights(text, **SCENES[scene]())
    with torch.no_grad():
        return te.encode_token_weights(t)


def reset(calls):
    calls.update(visual=0, llm_tokens=0)


def assert_same(a, b, exact=False):
    assert a[0].shape == b[0].shape
    if exact:
        assert torch.equal(a[0], b[0])
    else:
        torch.testing.assert_close(b[0], a[0], rtol=1e-4, atol=1e-5)
    assert torch.equal(a[2]["minimax_token_tags"], b[2]["minimax_token_tags"])
    assert a[2]["minimax_token_tags"].shape[0] == a[0].shape[1]


def test_cache_matches_uncached_and_reuses_prefix(tiny, monkeypatch):
    te, clip, tok, calls = tiny
    scenes = ["img1", "img2"] if clip.layer == "last" and type(clip.transformer) is minimax.MiniMaxQwen3VL else ["img2"]
    if type(clip.transformer) is minimax.MiniMaxQwen3VL:
        scenes.append("ref2va")
    prompts = ["make the sky purple and keep the logo", "a calm lake at dawn, photo"]
    for scene in scenes:
        monkeypatch.setattr(minimax, "PREFIX_CACHE", False)
        plain = [encode(te, tok, p, scene) for p in prompts]
        assert clip.prefix_cache_status == "off"
        monkeypatch.setattr(minimax, "PREFIX_CACHE", True)

        reset(calls)
        first = encode(te, tok, prompts[0], scene)
        assert clip.prefix_cache_status == "miss"
        assert calls["visual"] > 0
        total = first[0].shape[1]
        assert calls["llm_tokens"] == total                      # prefix pass + suffix pass: every token once
        assert_same(plain[0], first)

        prefix_len = clip._prefix_cache["prefix_len"]
        reset(calls)
        second = encode(te, tok, prompts[1], scene)               # same content in new tensors, a different prompt
        assert clip.prefix_cache_status == "hit"
        assert calls["visual"] == 0
        assert calls["llm_tokens"] == second[0].shape[1] - prefix_len
        assert calls["llm_tokens"] < second[0].shape[1]
        assert_same(plain[1], second)

        reset(calls)
        again = encode(te, tok, prompts[0], scene)                # back to the first prompt, still a hit
        assert clip.prefix_cache_status == "hit" and calls["visual"] == 0
        assert_same(plain[0], again)
        te_prefix = clip._prefix_cache
        assert te_prefix["states"].device.type == "cpu"
        clip._prefix_cache = None


def test_changed_inputs_miss(tiny):
    te, clip, tok, calls = tiny
    encode(te, tok, "one", "img2")
    assert clip.prefix_cache_status == "miss"
    encode(te, tok, "two", "img2")
    assert clip.prefix_cache_status == "hit"
    for scene in ("img2_edited", "img2_swapped"):  # one image's pixels / the image order
        encode(te, tok, "two", scene)
        assert clip.prefix_cache_status == "miss", scene
        encode(te, tok, "three", scene)
        assert clip.prefix_cache_status == "hit", scene
    encode(te, tok, "three", "img2")                # one entry only: the old content is gone
    assert clip.prefix_cache_status == "miss"

    # the weight-patch signal of the wrapper (ModelPatcher's current_weight_patches_uuid)
    encode(te, tok, "three", "img2")
    assert clip.prefix_cache_status == "hit"
    te.current_weight_patches_uuid = uuid.uuid4()
    encode(te, tok, "three", "img2")
    assert clip.prefix_cache_status == "miss"
    encode(te, tok, "four", "img2")
    assert clip.prefix_cache_status == "hit"
    te.current_weight_patches_uuid = uuid.uuid4()
    encode(te, tok, "four", "img2")
    assert clip.prefix_cache_status == "miss"


def test_fallbacks_are_identical(tiny, monkeypatch):
    te, clip, tok, calls = tiny
    cases = [("a plain text prompt", dict(images=[])), ("", SCENES["img2"]())]
    for text, kw in cases:
        t = tok.tokenize_with_weights(text, **kw)
        monkeypatch.setattr(minimax, "PREFIX_CACHE", False)
        with torch.no_grad():
            plain = te.encode_token_weights(t)
        monkeypatch.setattr(minimax, "PREFIX_CACHE", True)
        with torch.no_grad():
            cached = te.encode_token_weights(t)
        assert clip.prefix_cache_status == "fallback"
        assert clip._prefix_cache is None
        assert_same(plain, cached, exact=True)


def test_weight_hooks_bypass_cache(tiny):
    # CLIP hooks patch weights in place without a new patches uuid (sd.py passes weight_hooks)
    te, clip, tok, calls = tiny
    encode(te, tok, "one", "img2")
    assert clip.prefix_cache_status == "miss"
    te.set_clip_options({"weight_hooks": True})
    hooked = encode(te, tok, "two", "img2")
    assert clip.prefix_cache_status == "fallback"
    te.reset_clip_options()
    assert not clip.weight_hooks
    plain = encode(te, tok, "two", "img2")
    assert clip.prefix_cache_status == "hit"
    assert_same(hooked, plain)


def test_vision_runs_fp32(tiny32, monkeypatch):
    # bf16 vision changed the 32B output far beyond rounding on real weights: the tower stays fp32
    te, clip, tok, calls = tiny32
    seen = []
    patch_embed = clip.transformer.visual.patch_embed.forward
    def spy(x, *a, **k):
        seen.append(x.dtype)
        return patch_embed(x, *a, **k)
    clip.transformer.visual.patch_embed.forward = spy

    clip.transformer.visual.to(torch.bfloat16)
    clip._prefix_cache = None
    monkeypatch.setattr(minimax, "PREFIX_CACHE", False)
    plain = encode(te, tok, "bf16 weights", "ref2va")
    assert seen and set(seen) == {torch.float32}               # images and video blocks
    monkeypatch.setattr(minimax, "PREFIX_CACHE", True)
    cached = encode(te, tok, "bf16 weights", "ref2va")
    assert clip.prefix_cache_status == "miss"
    assert plain[0].dtype == torch.float32 and torch.isfinite(plain[0]).all()
    torch.testing.assert_close(cached[0], plain[0], rtol=1e-3, atol=1e-4)
    hit = encode(te, tok, "bf16 weights", "ref2va")
    assert clip.prefix_cache_status == "hit"
    torch.testing.assert_close(hit[0], plain[0], rtol=1e-3, atol=1e-4)
