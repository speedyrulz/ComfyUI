import numbers

import torch

import comfy.text_encoders.qwen3vl
from comfy import sd1_clip

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM_PROMPT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
T2I_TEMPLATE = SYSTEM_PROMPT + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class QwenImage21Tokenizer(comfy.text_encoders.qwen3vl.Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_8b")
        self.llama_template = T2I_TEMPLATE

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, keep_vision=False, prefix_share=None, **kwargs):
        image = kwargs.get("image", None)
        if image is not None and len(images) == 0:
            images = [image[i:i + 1] for i in range(image.shape[0])]
        if llama_template is None and len(images) > 0:
            refs = " ".join("<image{}>{}".format(i + 1, VISION_BLOCK) for i in range(len(images)))
            llama_template = T2I_TEMPLATE.replace("{}", refs + "{}", 1)
        out = super().tokenize_with_weights(text, return_word_ids=return_word_ids, llama_template=llama_template, images=images, prevent_empty_text=prevent_empty_text, **kwargs)
        out["keep_vision"] = keep_vision
        if prefix_share is not None:
            out["prefix_share"] = prefix_share
        return out


class QwenImage21Qwen3VLClipModel(comfy.text_encoders.qwen3vl.Qwen3VLClipModel):
    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, dtype=dtype, attention_mask=attention_mask, model_options=model_options, model_type="qwen3vl_8b")
        # last layer without the final RMSNorm: transformers 4.57 hidden_states[-1], which Qwen's results are tuned to (5.x norms it)
        self.layer_norm_hidden_state = False
        self.image_spans = []
        self.prefix_share = None  # a dict shared by prompts with the same reference images (see forward)

    def process_tokens(self, tokens, device):
        embeds, attention_mask, num_tokens, embeds_info = super().process_tokens(tokens, device)
        self.image_spans = [(e["index"], e["size"]) for e in embeds_info if e["type"] == "image"]
        return embeds, attention_mask, num_tokens, embeds_info

    def forward(self, tokens):
        # Positive and negative share everything up to the last reference image (system turn, user header, images):
        # with a causal LM that prefix's hidden states and keys/values are the same for both, so run the vision
        # tower once, keep the prefix's KV cache, and for every later prompt encode only its own text after it.
        share = self.prefix_share
        if share is None or len(tokens) != 1 or self.layer not in ("hidden", "last") or not isinstance(self.layer_idx, int):
            return super().forward(tokens)

        device = self.execution_device
        if device is None:
            device = self.transformer.get_input_embeddings().weight.device

        self.transformer.vision_cache = share.setdefault("vision", {})
        try:
            embeds, attention_mask, num_tokens, embeds_info = self.process_tokens(tokens, device)
        finally:
            self.transformer.vision_cache = None

        images = [e for e in embeds_info if e.get("type") == "image"]
        prefix_len = max((e["index"] + e["size"] for e in images), default=0)
        if prefix_len == 0 or prefix_len >= embeds.shape[1] or (attention_mask is not None and not bool(attention_mask.all())):
            return self.finish_forward(self.run_transformer(embeds, attention_mask, embeds_info), attention_mask)

        # the prefix as the tokenizer wrote it: ids, and each image / embedding by identity
        last = max(i for i, t in enumerate(tokens[0]) if isinstance(t, dict) and t.get("type") == "image")
        head = tokens[0][:last + 1]
        key = tuple(t if isinstance(t, numbers.Integral) else id(t.get("data", t) if isinstance(t, dict) else t) for t in head)

        position_ids, visual_pos_masks, deepstack = self.transformer.build_image_inputs(embeds, embeds_info)
        prefix = share.get("prefix")
        if prefix is None or prefix["key"] != key or prefix["device"] != embeds.device:
            cfg = self.transformer.model.config
            shape = (1, cfg.num_key_value_heads, prefix_len, cfg.head_dim)
            past = [(torch.empty(shape, device=embeds.device, dtype=embeds.dtype), torch.empty(shape, device=embeds.device, dtype=embeds.dtype), 0)
                    for _ in range(self.transformer.num_layers)]
            out = self.run_transformer(embeds[:, :prefix_len], None, embeds_info, position_ids=position_ids[..., :prefix_len],
                                       visual_pos_masks=visual_pos_masks[:, :prefix_len], deepstack_embeds=deepstack, past_key_values=past)
            prefix = {"key": key, "device": embeds.device, "x": out[0], "intermediate": out[1], "kv": out[2],
                      "refs": [t for t in head if not isinstance(t, numbers.Integral)]}  # keeps the ids in key from being reused
            share["prefix"] = prefix

        # the cached KV is exactly prefix_len long, so the attention concatenates onto it and never writes into it
        out = self.run_transformer(embeds[:, prefix_len:], None, embeds_info, position_ids=position_ids[..., prefix_len:],
                                   visual_pos_masks=None, deepstack_embeds=None, past_key_values=prefix["kv"])
        x = torch.cat((prefix["x"], out[0]), dim=1)
        intermediate = None if out[1] is None else torch.cat((prefix["intermediate"], out[1]), dim=1)
        return self.finish_forward((x, intermediate), attention_mask)

    def run_transformer(self, embeds, attention_mask, embeds_info, **kwargs):
        attention_mask_model = attention_mask if self.enable_attention_masks else None
        return self.transformer(None, attention_mask_model, embeds=embeds, num_tokens=None, intermediate_output=self.layer_idx,
                                final_layer_norm_intermediate=self.layer_norm_hidden_state, dtype=torch.float32, embeds_info=embeds_info, **kwargs)

    def finish_forward(self, outputs, attention_mask):
        # the tail of SDClipModel.forward for a "last" or "hidden" layer
        z = outputs[0].float() if self.layer == "last" else outputs[1].float()
        if self.zero_out_masked:
            z *= attention_mask.unsqueeze(-1).float()
        if self.return_attention_masks:
            return z, None, {"attention_mask": attention_mask}
        return z, None


class QwenImage21TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_8b", clip_model=QwenImage21Qwen3VLClipModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        clip = getattr(self, self.clip)
        clip.prefix_share = token_weight_pairs.get("prefix_share")
        try:
            out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        finally:
            clip.prefix_share = None
        tokens = [t[0] for t in token_weight_pairs["qwen3vl_8b"][0]]
        image_spans = clip.image_spans

        # drop the system turn, everything before the second <|im_start|>; positions shift by each image expanded before it
        im_starts, offset, spans = [], 0, iter(image_spans)
        for i, t in enumerate(tokens):
            if isinstance(t, numbers.Integral):
                if t == 151644:
                    im_starts.append(i + offset)
            elif isinstance(t, dict) and t.get("type") == "image":  # a textual embedding is a bare tensor and has no span
                offset += next(spans, (0, 1))[1] - 1
        keep = torch.ones(out.shape[1], dtype=torch.bool)
        keep[:im_starts[1] if len(im_starts) > 1 else 0] = False

        # vision tokens are replaced by reference latents in the DiT: drop them and record where each image goes
        # with no latents coming (no vae) they stay, and the image conditions through the text encoder alone
        slots = []
        if not token_weight_pairs.get("keep_vision", False):
            for start, size in image_spans:
                keep[start:start + size] = False
                slots.append(int(keep[:start].sum()))

        out = out[:, keep.to(out.device)]
        extra["attention_mask"] = extra["attention_mask"][:, keep.to(extra["attention_mask"].device)]
        if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
            extra.pop("attention_mask")
        if len(slots) > 0:
            extra["image_slots"] = slots
        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class QwenImage21TEModel_(QwenImage21TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return QwenImage21TEModel_
