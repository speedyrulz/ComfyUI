import numbers

import comfy.text_encoders.qwen35
from comfy import sd1_clip

# PAN 2 conditions on hidden_states[-3] of a stock Qwen3.5-9B, on the tokens after the `<|im_start|>user\n` header:
# the prompt and the assistant header (with its open think block) that the chat template adds after it.
SYSTEM_PROMPT = "You are a helpful assistant. Generate the video with the following instructions."
TEMPLATE = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
MAX_SEQUENCE_LENGTH = 1536
IM_START = 248045
USER_HEADER_LENGTH = 3  # <|im_start|> user \n


class PAN2Tokenizer(comfy.text_encoders.qwen35.Qwen35ImageTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen35_9b")

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, **kwargs):
        if llama_template is None and not text.startswith("<|im_start|>"):
            # the chat template trims the user turn (the reference's single space for an empty prompt ends up empty)
            text, llama_template = text.strip(), TEMPLATE
        return super().tokenize_with_weights(text, return_word_ids=return_word_ids, llama_template=llama_template, **kwargs)


class PAN2TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, mtp=False):
        clip_model = lambda **kw: comfy.text_encoders.qwen35.Qwen35ClipModel(**kw, layer="hidden", layer_idx=-3, model_type="qwen35_9b", mtp=mtp)
        super().__init__(device=device, dtype=dtype, name="qwen35_9b", clip_model=clip_model, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        tokens = [t[0] for t in token_weight_pairs["qwen35_9b"][0]]
        starts = [i for i, t in enumerate(tokens) if isinstance(t, numbers.Integral) and t == IM_START]
        # the user turn is the second one of the template; a prompt that brings its own template may have only one
        start = (starts[1] if len(starts) > 1 else starts[0]) + USER_HEADER_LENGTH if len(starts) > 0 else 0
        out = out[:, start:start + MAX_SEQUENCE_LENGTH]
        extra.pop("attention_mask", None)  # one prompt is one sequence, there is no padding
        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None, mtp=False):
    class PAN2TEModel_(PAN2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options, mtp=mtp)
    return PAN2TEModel_
