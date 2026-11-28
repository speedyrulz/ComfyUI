"""MiniMax H3 text/vision conditioning: Qwen3-VL-32B (truncated to 50 layers).

The H3 presentation is NOT chat-templated: token ids are raw prompt/label text
(no special tokens) with explicit vision blocks spliced in:

  t2va:   <prompt>
  fl2va:  "<Picture 1>: " <vision block> ["<Picture 2>: " <vision block>] <prompt>
  image:  "Picture 1: " <vision block> ["Picture 2: " <vision block> ...] <prompt>  (MiniMax-H3-Image edits)
  ref2va: per condition in request order (1-based ordinals per type):
            image -> "<Picture i>: " <vision block>
            audio -> "<Audio j>: "              (audio never enters Qwen)
            video -> "<Video k>: " then per 2-frame temporal block
                     "<T.T seconds>" <vision block(2 frames)>
          then <prompt>

The conditioning is the unnormalized hidden state after LM layer 50 (the
converted checkpoint is truncated there, so this is simply the last-layer
output with no final norm). Vision-pad positions carry adaLN token tag 0
(video modality) in the DiT; text positions carry tag 1 — the tags are
returned alongside the embeddings as "minimax_token_tags".
"""

import hashlib
import logging
import numbers
import os

import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.ops
import comfy.sd1_clip
from .qwen3vl import Qwen3VL, Qwen3VLSDTokenizer
from .qwen_vl import smart_resize, process_qwen2vl_images

# Persistent prefix cache (see MiniMaxH3ClipModel.forward). Set COMFY_MINIMAX_PREFIX_CACHE=0 to disable.
PREFIX_CACHE = os.environ.get("COMFY_MINIMAX_PREFIX_CACHE", "1") != "0"

VISION_START = 151652
VISION_END = 151653
# FL2VA/Ref2VA tokenizer_config extends Qwen with these, ids fixed by the released tokenizer
MINIMAX_EXTRA_TOKENS = {"<d>": 151669, "</d>": 151670, "<|cutoff|>": 151671,
                        "<|lyrics_start|>": 151672, "<|lyrics_end|>": 151673,
                        "<|caption_start|>": 151674, "<|caption_end|>": 151675}
QWEN_IMAGE_MEAN = [0.5, 0.5, 0.5]
QWEN_IMAGE_STD = [0.5, 0.5, 0.5]
# p-h3-image: Qwen3-VL-4B hidden states read by the text connector, the last one normed
PH3_IMAGE_HIDDEN_STATES = [13, 19, 25, 31, 36]
PH3_IMAGE_TEXT_DIM = 5120


def process_video_block(frames, patch_size=16, temporal_patch_size=2, merge_size=2,
                        min_pixels=3136, max_pixels=12845056):
    """[2, H, W, C] frame pair -> (flatten_patches, grid_thw) with grid_t=1.

    Same resize/normalize policy as process_qwen2vl_images, but the two frames
    fill the temporal patch instead of repeating a single frame.
    """
    t, height, width, _ = frames.shape
    imgs = frames.permute(0, 3, 1, 2)
    h_bar, w_bar = smart_resize(height, width, patch_size * merge_size, min_pixels, max_pixels)

    imgs = F.interpolate(imgs, size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    mean = torch.tensor(QWEN_IMAGE_MEAN, device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(QWEN_IMAGE_STD, device=imgs.device).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    patches = imgs.reshape(1, temporal_patch_size, 3, grid_h // merge_size, merge_size,
                           patch_size, grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    grid_thw = torch.stack([torch.tensor([1, grid_h, grid_w], device=frames.device, dtype=torch.long)])
    return flatten, grid_thw


def token_tags_from_embeds_info(seq_len, embeds_info):
    # whole vision block VIDEO(0), including the flanking <|vision_start|>/<|vision_end|> tokens
    # embeds_info spans cover only the expanded embeddings, so widen by one on each side.
    tags = torch.ones(seq_len, dtype=torch.long)
    for e in embeds_info:
        if e.get("type") == "image":
            tags[max(0, e["index"] - 1):e["index"] + e["size"] + 1] = 0
    return tags


class MiniMaxQwen3VL(Qwen3VL):
    model_type = "qwen3vl_32b"

    def vision_dtype(self):
        # bf16 vision weights run the tower in bf16 (as the transformers reference does); anything else (fp16 can
        # overflow in the tower, fp32) stays fp32. The text-encoder ops cast weights to the input dtype.
        if self.visual.patch_embed.proj.weight.dtype == torch.bfloat16:
            return torch.bfloat16
        return torch.float32

    def preprocess_embed(self, embed, device):
        if embed["type"] != "image":
            return None, None
        data = embed["data"]
        # optional {id(data): (data, merged, extra)} set by a caller that encodes the same images more than once
        cache = self.vision_cache
        if cache is not None:
            hit = cache.get(id(data))
            if hit is not None and hit[0] is data:
                return hit[1], hit[2]
        if embed.get("minimax_video_block", False):
            flatten, grid = process_video_block(data)
        else:
            # Qwen3-VL normalizes to [-1, 1] (mean/std 0.5)
            flatten, grid = process_qwen2vl_images(data, patch_size=16, image_mean=QWEN_IMAGE_MEAN, image_std=QWEN_IMAGE_STD)
        merged, deepstack = self.visual(flatten.to(device, dtype=self.vision_dtype()), grid)
        extra = {"grid": grid, "deepstack": deepstack}
        if cache is not None:
            cache[id(data)] = (data, merged, extra)
        return merged, extra

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None,
                intermediate_output=None, final_layer_norm_intermediate=True,
                dtype=None, embeds_info=[], **kwargs):
        seq = embeds.shape[1] if embeds is not None else input_ids.shape[1]
        self.last_token_tags = token_tags_from_embeds_info(seq, embeds_info)
        return super().forward(input_ids, attention_mask=attention_mask, embeds=embeds,
                               num_tokens=num_tokens, intermediate_output=intermediate_output,
                               final_layer_norm_intermediate=final_layer_norm_intermediate,
                               dtype=dtype, embeds_info=embeds_info, **kwargs)


class TextConnector(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.input_mean = torch.nn.Parameter(torch.empty(in_channels, dtype=dtype, device=device))
        self.input_std = torch.nn.Parameter(torch.empty(in_channels, dtype=dtype, device=device))
        self.proj = operations.Linear(in_channels, out_channels, bias=False, dtype=dtype, device=device)
        self.output_mean = torch.nn.Parameter(torch.empty(out_channels, dtype=dtype, device=device))
        self.output_std = torch.nn.Parameter(torch.empty(out_channels, dtype=dtype, device=device))
        self.sink_embedding = torch.nn.Parameter(torch.empty(out_channels, dtype=dtype, device=device))

    def forward(self, hidden_states):
        # [B, layers, T, C] -> [B, T, layers * C] standardized, projected and rescaled; the first token is replaced by the sink
        x = hidden_states.movedim(1, 2).flatten(2)
        x = (x - comfy.ops.cast_to_input(self.input_mean, x)) / comfy.ops.cast_to_input(self.input_std, x)
        x = self.proj(x)
        x = torch.addcmul(comfy.ops.cast_to_input(self.output_mean, x), x, comfy.ops.cast_to_input(self.output_std, x))
        sink = comfy.ops.cast_to_input(self.sink_embedding, x).expand(x.shape[0], 1, -1)
        return torch.cat([sink, x[:, 1:]], dim=1)


class PH3ImageQwen3VL(MiniMaxQwen3VL):
    model_type = "qwen3vl_4b"

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__(config_dict, dtype, device, operations)
        # float32 and never quantized: the mixed-precision ops of a quantized file would load it at the compute dtype
        self.text_connector = TextConnector(len(PH3_IMAGE_HIDDEN_STATES) * self.model.config.hidden_size, PH3_IMAGE_TEXT_DIM,
                                            dtype=torch.float32, device=device, operations=comfy.ops.manual_cast)

    def forward(self, *args, apply_connector=True, **kwargs):
        out = super().forward(*args, **kwargs)
        if not apply_connector:  # the prefix cache joins prefix and suffix states first: the connector replaces token 0 with a sink
            return out
        return (out[0], self.text_connector(out[1])) + tuple(out[2:])


class MiniMaxH3ClipModel(comfy.sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_class=MiniMaxQwen3VL, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=None, textmodel_json_config={},
                         dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                         model_class=model_class, enable_attention_masks=False,
                         return_attention_masks=False, model_options=model_options)

        self.patch_state = None  # the weight-patch signal of the owning TE wrapper (see PatchStateMixin)
        self.weight_hooks = False  # CLIP hooks patch weights without changing patch_state: no prefix cache then
        self.prefix_cache_status = "off"  # "hit", "miss", "fallback" or "off" for the last encode
        self._prefix_cache = None  # the single entry, see forward

    def set_clip_options(self, options):
        super().set_clip_options(options)
        self.weight_hooks = options.get("weight_hooks", self.weight_hooks)

    def reset_clip_options(self):
        super().reset_clip_options()
        self.weight_hooks = False

    def load_sd(self, sd):
        self._prefix_cache = None
        return super().load_sd(sd)

    # ---- persistent prefix cache -------------------------------------------------------------------------------
    # Every H3 layout puts the reference vision blocks first and the prompt last, and the LM is causal, so the hidden
    # states / intermediates / KV of everything up to the last <|vision_end|> do not depend on the prompt. One entry,
    # keyed by content (the conditioning nodes rebuild the image tensors on every run), survives between executions:
    # re-encoding the same references with a new prompt skips the vision tower and the prefix LM pass.
    # Size, kept on intermediate_device() (CPU) between encodes: fp32 KV is 2 * kv_heads * head_dim * layers * 4 bytes
    # per prefix token (32B: 8 * 128 * 50 -> ~0.4 MB/token, so ~1.2 GB for three 1k-token images), plus 20 KB/token
    # (32B last state) or 51 KB/token (PH3: 5 raw intermediates) of outputs.

    @staticmethod
    def _hash_tensor(t, h):
        t = t.detach().contiguous().cpu()
        h.update(repr((tuple(t.shape), str(t.dtype))).encode())
        if t.numel() > 0:
            h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))

    def _prefix_key(self, head):
        # content key of the prefix tokens, or None when an entry is not something we can hash
        h = hashlib.blake2b(digest_size=32)
        h.update(repr((type(self.transformer).__name__, repr(self.layer), str(self.patch_state))).encode())
        for t in head:
            if isinstance(t, numbers.Integral):
                h.update(b"i%d;" % int(t))
                continue
            if isinstance(t, dict):
                data = t.get("data")
                h.update(("d:%s:%s:" % (t.get("type"), bool(t.get("minimax_video_block", False)))).encode())
            else:
                data = t
                h.update(b"t:")
            if not torch.is_tensor(data):
                return None
            self._hash_tensor(data, h)
        return h.digest()

    def _prefix_end(self, tokens):
        # index of the <|vision_end|> after the last vision block, or None when the prefix cache does not apply
        if len(tokens) != 1 or not (self.layer == "last" or isinstance(self.layer, list)):
            return None
        entries = tokens[0]
        last = max((i for i, t in enumerate(entries) if isinstance(t, dict) and t.get("type") == "image"), default=-1)
        if last < 0 or last + 2 >= len(entries):
            return None
        end = entries[last + 1]
        if not isinstance(end, numbers.Integral) or int(end) != VISION_END:
            return None
        if not any(isinstance(t, numbers.Integral) for t in entries[last + 2:]):
            return None
        return last + 1

    def _run(self, embeds, embeds_info, raw=False, **kwargs):
        if raw and hasattr(self.transformer, "text_connector"):
            kwargs["apply_connector"] = False
        intermediate_output = self.layer if isinstance(self.layer, list) else self.layer_idx
        return self.transformer(None, None, embeds=embeds, intermediate_output=intermediate_output,
                                final_layer_norm_intermediate=self.layer_norm_hidden_state, dtype=torch.float32,
                                embeds_info=embeds_info, **kwargs)

    def _plain(self, tokens, status):
        self.prefix_cache_status = status
        return super().forward(tokens)

    def forward(self, tokens):
        if not PREFIX_CACHE:
            self._prefix_cache = None
            return self._plain(tokens, "off")
        end = None if self.weight_hooks else self._prefix_end(tokens)
        if end is None:
            return self._plain(tokens, "fallback")
        head = tokens[0][:end + 1]
        key = self._prefix_key(head)
        if key is None:
            return self._plain(tokens, "fallback")

        device = self.execution_device
        if device is None:
            device = self.transformer.get_input_embeddings().weight.device
        store_device = comfy.model_management.intermediate_device()
        entry = self._prefix_cache
        if entry is not None and entry["key"] != key:
            entry = self._prefix_cache = None  # release the old KV before the new one is built

        if entry is not None:  # hit: no vision tower, no prefix pass; only the prompt text is embedded and run
            suffix = tokens[0][end + 1:]
            embeds, _, _, embeds_info = self.process_tokens([suffix], device)
            n = embeds.shape[1]
            first = entry["first_pos"]
            position_ids = torch.arange(first, first + n, device=device, dtype=torch.float32).unsqueeze(0).repeat(3, 1)
            kv = [(k.to(device), v.to(device), entry["prefix_len"]) for k, v in entry["kv"]]
            out = self._run(embeds, embeds_info, raw=True, position_ids=position_ids, past_key_values=kv)
            prefix = entry["states"].to(device)
            status = "hit"
            prefix_tags = entry["tags"]
            seq = entry["prefix_len"] + n
        else:  # miss: the whole sequence's inputs, then the prefix pass (filling an exact-size KV) and the suffix pass
            embeds, _, num_tokens, embeds_info = self.process_tokens(tokens, device)
            images = [e for e in embeds_info if e.get("type") == "image"]
            n_vision = sum(1 for t in head if isinstance(t, dict) and t.get("type") == "image")
            prefix_len = images[-1]["index"] + images[-1]["size"] + 1 if images else 0
            if len(images) != n_vision or prefix_len == 0 or prefix_len >= embeds.shape[1]:
                self.prefix_cache_status = "fallback"
                outputs = self._run(embeds, embeds_info, num_tokens=num_tokens)
                z = (outputs[0] if self.layer == "last" else outputs[1]).float()
                return z, None
            seq = embeds.shape[1]
            position_ids, visual_pos_masks, deepstack = self.transformer.build_image_inputs(embeds, embeds_info)
            cfg = self.transformer.model.config
            shape = (1, cfg.num_key_value_heads, prefix_len, cfg.head_dim)
            past = [(torch.empty(shape, device=embeds.device, dtype=embeds.dtype), torch.empty(shape, device=embeds.device, dtype=embeds.dtype), 0)
                    for _ in range(self.transformer.num_layers)]
            pre = self._run(embeds[:, :prefix_len], embeds_info, raw=True, position_ids=position_ids[..., :prefix_len],
                            visual_pos_masks=visual_pos_masks[:, :prefix_len], deepstack_embeds=deepstack, past_key_values=past)
            kv = pre[2]  # exactly prefix_len long: the suffix attention concatenates onto it and never writes into it
            out = self._run(embeds[:, prefix_len:], embeds_info, raw=True, position_ids=position_ids[..., prefix_len:], past_key_values=kv)
            prefix = pre[0] if self.layer == "last" else pre[1]
            status = "miss"
            prefix_tags = token_tags_from_embeds_info(seq, embeds_info)[:prefix_len]
            # text after the last vision block is sequential and equal on the three axes: one number carries it
            self._prefix_cache = {"key": key, "prefix_len": prefix_len, "first_pos": int(position_ids[0, prefix_len].item()),
                                  "states": prefix.to(store_device), "tags": prefix_tags,
                                  "kv": [(k.to(store_device), v.to(store_device)) for k, v, _ in kv]}

        suffix_states = out[0] if self.layer == "last" else out[1]
        if self.layer == "last":
            z = torch.cat((prefix, suffix_states), dim=1)
        else:  # [B, layers, T, C] raw intermediates, joined before the connector (it replaces token 0 with a sink)
            z = torch.cat((prefix, suffix_states), dim=2)
            if hasattr(self.transformer, "text_connector"):
                z = self.transformer.text_connector(z)
        self.prefix_cache_status = status
        self.transformer.last_token_tags = torch.cat((prefix_tags, torch.ones(seq - prefix_tags.shape[0], dtype=torch.long)))
        return z.float(), None

    def encode_token_weights(self, token_weight_pairs):
        out = super().encode_token_weights(token_weight_pairs)
        tags = getattr(self.transformer, "last_token_tags", None)
        if tags is not None:
            extra = out[2] if len(out) > 2 and isinstance(out[2], dict) else {}
            extra["minimax_token_tags"] = tags
            out = (out[0], out[1], extra)
        return out


class PH3ImageClipModel(MiniMaxH3ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, layer=PH3_IMAGE_HIDDEN_STATES, dtype=dtype, model_class=PH3ImageQwen3VL, model_options=model_options)


class PatchStateMixin:
    # ModelPatcher records the patch set it loaded (uuid, rerolled by every add_patches such as a CLIP LoRA) as
    # current_weight_patches_uuid on the model it wraps, which is this SD1ClipModel wrapper, not the inner clip model.
    # Hand it down so the prefix cache is invalidated when the patches change.
    def encode_token_weights(self, token_weight_pairs):
        getattr(self, self.clip).patch_state = getattr(self, "current_weight_patches_uuid", None)
        return super().encode_token_weights(token_weight_pairs)


class MiniMaxH3TEModel(PatchStateMixin, comfy.sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_32b",
                         clip_model=MiniMaxH3ClipModel, model_options=model_options)


class PH3ImageTEModel(PatchStateMixin, comfy.sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_4b",
                         clip_model=PH3ImageClipModel, model_options=model_options)


class MiniMaxQwenSDTokenizer(Qwen3VLSDTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer.add_special_tokens({"additional_special_tokens": list(MINIMAX_EXTRA_TOKENS)})
        self.inv_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}


class MiniMaxH3Tokenizer(comfy.sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = lambda *a, **kw: MiniMaxQwenSDTokenizer(*a, **kw, embedding_size=5120, embedding_key="qwen3vl_32b")
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen3vl_32b", tokenizer=tokenizer)

    @staticmethod
    def _vision_entry(data, video_block=False):
        emb = {"type": "image", "data": data, "original_type": "image"}
        if video_block:
            emb["minimax_video_block"] = True
        return emb

    def tokenize_with_weights(self, text, return_word_ids=False, images=[],
                              minimax_ref_items=None, picture_label="<Picture %d>: ", **kwargs):
        entries = []

        def add_text(s):
            if not s:
                return
            token_batches = getattr(self, self.clip).tokenize_with_weights(
                s,
                return_word_ids=False,
                disable_weights=True,
            )
            if len(token_batches) != 1:
                raise ValueError("MiniMax H3 text segment exceeds the supported prompt length.")
            entries.extend(token_batches[0])

        def add_vision(data, video_block=False):
            entries.append((VISION_START, 1.0))
            entries.append((self._vision_entry(data, video_block), 1.0))
            entries.append((VISION_END, 1.0))

        if minimax_ref_items:
            counters = {"image": 0, "audio": 0, "video": 0}
            for item in minimax_ref_items:
                kind = item["type"]
                counters[kind] += 1
                if kind == "image":
                    add_text("<Picture %d>: " % counters["image"])
                    add_vision(item["data"])
                elif kind == "audio":
                    add_text("<Audio %d>: " % counters["audio"])
                elif kind == "video":
                    frames = item["data"]  # [T, H, W, C], sampled at 2 fps
                    timestamps = item.get("timestamps")
                    if timestamps is None:
                        timestamps = [i / 2.0 for i in range(frames.shape[0])]
                    if frames.shape[0] % 2 == 1:  # repeat-pad to temporal patch of 2
                        frames = torch.cat([frames, frames[-1:]], dim=0)
                        timestamps = list(timestamps) + [timestamps[-1]]
                    add_text("<Video %d>: " % counters["video"])
                    for i in range(0, frames.shape[0], 2):
                        block_ts = (timestamps[i] + timestamps[i + 1]) / 2.0
                        add_text("<%.1f seconds>" % block_ts)
                        add_vision(frames[i:i + 2], video_block=True)
        else:
            for i, img in enumerate(images):
                add_text(picture_label % (i + 1))
                add_vision(img)

        add_text(text)
        if len(entries) == 0:
            entries.append((151643, 1.0))
        if return_word_ids:
            entries = [t + (0,) for t in entries]
        return {self.clip: [entries]}


class PH3ImageTokenizer(MiniMaxH3Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = lambda *a, **kw: Qwen3VLSDTokenizer(*a, **kw, embedding_size=2560, embedding_key="qwen3vl_4b")
        comfy.sd1_clip.SD1Tokenizer.__init__(self, embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen3vl_4b", tokenizer=tokenizer)


def te(dtype_llama=None, llama_quantization_metadata=None, te_model=MiniMaxH3TEModel, **kwargs):
    class MiniMaxH3TEModel_(te_model):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return MiniMaxH3TEModel_
