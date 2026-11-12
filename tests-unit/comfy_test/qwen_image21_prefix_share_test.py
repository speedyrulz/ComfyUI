from dataclasses import dataclass

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.text_encoders.llama as llama  # noqa: E402
import comfy.text_encoders.qwen3vl as qwen3vl  # noqa: E402
import comfy.text_encoders.qwen_image21 as qwen_image21  # noqa: E402


@dataclass
class _TinyConfig(llama.Qwen3VL_8BConfig):
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 32
    lm_head: bool = False
    rope_dims = [8, 4, 4]


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setitem(qwen3vl.QWEN3VL_CONFIGS, "qwen3vl_8b", _TinyConfig)
    monkeypatch.setitem(qwen3vl.QWEN3VL_VISION, "qwen3vl_8b", dict(hidden_size=128, intermediate_size=128, depth=3, deepstack_visual_indexes=[0, 1, 2]))
    torch.manual_seed(0)
    te = qwen_image21.te()(device="cpu", dtype=torch.float32)
    with torch.no_grad():
        for name, p in te.named_parameters():
            p.normal_(0, 0.02)
            if "norm" in name and name.endswith("weight"):
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
    return te, qwen_image21.QwenImage21Tokenizer(), calls


def _encode(te, tok, images, prompts, share, keep_vision):
    out = []
    for text in prompts:
        t = tok.tokenize_with_weights(text, images=images, keep_vision=keep_vision, prevent_empty_text=True, prefix_share=share)
        with torch.no_grad():
            out.append(te.encode_token_weights(t))
    return out


def _assert_same(a, b):
    for (ca, _, ea), (cb, _, eb) in zip(a, b):
        assert ca.shape == cb.shape
        torch.testing.assert_close(cb, ca, rtol=1e-4, atol=1e-5)
        assert ea.keys() == eb.keys()
        for k in ea:
            assert torch.equal(ea[k], eb[k]) if torch.is_tensor(ea[k]) else ea[k] == eb[k]


@pytest.mark.parametrize("keep_vision", [False, True])
@pytest.mark.parametrize("negative", ["", "blurry, low quality, watermark"])
def test_shared_prefix_matches_plain_encode(tiny, keep_vision, negative):
    te, tok, calls = tiny
    images = [torch.rand(1, 96, 128, 3), torch.rand(1, 64, 64, 3)]
    prompts = ("make the sky purple, keep <image2> as the logo", negative)

    plain = _encode(te, tok, images, prompts, None, keep_vision)
    plain_calls = dict(calls)
    calls.update(visual=0, llm_tokens=0)
    shared = _encode(te, tok, images, prompts, {}, keep_vision)

    _assert_same(plain, shared)
    assert plain_calls["visual"] == 4 and calls["visual"] == 2  # the vision tower runs once per image, not per prompt
    assert calls["llm_tokens"] < plain_calls["llm_tokens"]      # the negative skips the image prefix


def test_new_images_do_not_reuse_the_prefix(tiny):
    te, tok, _ = tiny
    images = [torch.rand(1, 64, 96, 3)]
    share = {}
    _encode(te, tok, images, ("one", "two"), share, False)
    images[0] = torch.rand(1, 64, 96, 3)
    _assert_same(_encode(te, tok, images, ("one", "two"), None, False), _encode(te, tok, images, ("one", "two"), share, False))
