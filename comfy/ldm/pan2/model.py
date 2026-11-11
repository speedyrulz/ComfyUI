# PAN 2 (IFM / MBZUAI) video diffusion transformer.
# Follows diffusers' PAN2Transformer3DModel and keeps its parameter names, so the diffusers checkpoint loads as it is.

import math

import torch
from torch import nn

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
from comfy.ldm.modules.attention import optimized_attention


def timestep_embedding(t, dim=256, max_period=10000):
    # diffusers Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0): [cos, sin]
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def rope_freqs(sizes, axes_dim, theta, temporal_scale, device):
    # (cos, sin) of shape [T * H * W, sum(axes_dim)]; only the frame positions are rescaled
    grids = torch.meshgrid(*[torch.arange(s, device=device, dtype=torch.float32) for s in sizes], indexing="ij")
    cos, sin = [], []
    for i, dim in enumerate(axes_dim):
        pos = grids[i].reshape(-1)
        if i == 0:
            pos = pos * temporal_scale
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2] / dim))
        freqs = torch.outer(pos, freqs)
        cos.append(freqs.cos().repeat_interleave(2, dim=1))
        sin.append(freqs.sin().repeat_interleave(2, dim=1))
    return torch.cat(cos, dim=1), torch.cat(sin, dim=1)


def apply_rope(x, pe):
    # x: [B, S, heads, head_dim]
    cos, sin = pe[0][None, :, None, :], pe[1][None, :, None, :]
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class FeedForward(nn.Module):
    # diffusers FeedForward: net.0.proj, activation, net.2
    def __init__(self, dim, inner_dim, swiglu=False, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.swiglu = swiglu
        proj = nn.Module()
        proj.proj = operations.Linear(dim, inner_dim * 2 if swiglu else inner_dim, bias=bias, dtype=dtype, device=device)
        self.net = nn.ModuleList([proj, nn.Identity(), operations.Linear(inner_dim, dim, bias=bias, dtype=dtype, device=device)])

    def forward(self, x):
        x = self.net[0].proj(x)
        if self.swiglu:
            x, gate = x.chunk(2, dim=-1)
            x = x * torch.nn.functional.silu(gate)
        else:
            x = torch.nn.functional.gelu(x, approximate="tanh")
        return self.net[2](x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, eps, bias=True, joint=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        kw = {"dtype": dtype, "device": device}
        self.to_q = operations.Linear(dim, dim, bias=bias, **kw)
        self.to_k = operations.Linear(dim, dim, bias=bias, **kw)
        self.to_v = operations.Linear(dim, dim, bias=bias, **kw)
        self.norm_q = operations.RMSNorm(dim_head, eps=eps, **kw)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, **kw)
        self.to_out = nn.ModuleList([operations.Linear(dim, dim, bias=bias, **kw)])
        if joint:
            self.add_q_proj = operations.Linear(dim, dim, bias=bias, **kw)
            self.add_k_proj = operations.Linear(dim, dim, bias=bias, **kw)
            self.add_v_proj = operations.Linear(dim, dim, bias=bias, **kw)
            self.norm_added_q = operations.RMSNorm(dim_head, eps=eps, **kw)
            self.norm_added_k = operations.RMSNorm(dim_head, eps=eps, **kw)
            self.to_add_out = operations.Linear(dim, dim, bias=bias, **kw)

    def forward(self, x, context=None, pe=None, transformer_options={}):
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        k = self.norm_k(self.to_k(x).unflatten(-1, (self.heads, -1)))
        v = self.to_v(x).unflatten(-1, (self.heads, -1))
        if pe is not None:  # only the video stream is rotated
            q, k = apply_rope(q, pe), apply_rope(k, pe)
        if context is not None:
            q = torch.cat([q, self.norm_added_q(self.add_q_proj(context).unflatten(-1, (self.heads, -1)))], dim=1)
            k = torch.cat([k, self.norm_added_k(self.add_k_proj(context).unflatten(-1, (self.heads, -1)))], dim=1)
            v = torch.cat([v, self.add_v_proj(context).unflatten(-1, (self.heads, -1))], dim=1)
        out = optimized_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), self.heads, skip_reshape=True, transformer_options=transformer_options)
        del q, k, v
        if context is None:
            return self.to_out[0](out)
        n = x.shape[1]
        return self.to_out[0](out[:, :n]), self.to_add_out(out[:, n:])


class TokenRefinerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_ratio, eps, dtype=None, device=None, operations=None):
        super().__init__()
        kw = {"dtype": dtype, "device": device, "operations": operations}
        self.norm1 = operations.RMSNorm(dim, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(dim, heads, dim_head, eps, bias=False, **kw)
        self.norm2 = operations.RMSNorm(dim, eps=eps, dtype=dtype, device=device)
        self.ff = FeedForward(dim, int(dim * mlp_ratio), swiglu=True, bias=False, **kw)

    def forward(self, x, transformer_options={}):
        x = x + self.attn(self.norm1(x), transformer_options=transformer_options)
        return x + self.ff(self.norm2(x))


class AdaLayerNorm(nn.Module):
    # holds the `linear` of diffusers' AdaLayerNormZero / AdaLayerNormContinuous
    def __init__(self, dim, chunks, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear = operations.Linear(dim, chunks * dim, dtype=dtype, device=device)

    def forward(self, temb):
        return self.linear(torch.nn.functional.silu(temb))


def layer_norm(x, eps=1e-6):
    return torch.nn.functional.layer_norm(x, x.shape[-1:], None, None, eps)


class TransformerBlock(nn.Module):
    # Joint attention over [video; text]; each stream has its own projections and feed-forward network. The video
    # stream is modulated by the timestep, the text stream by a learned table. As in the reference, the norms, their
    # shift/scale and the gated residuals run in float32.
    def __init__(self, dim, heads, dim_head, mlp_ratio, context_mlp_ratio, eps=1e-6, dtype=None, device=None, operations=None):
        super().__init__()
        kw = {"dtype": dtype, "device": device, "operations": operations}
        self.eps = eps
        self.norm1 = AdaLayerNorm(dim, 6, **kw)
        self.scale_shift_table_context = nn.Parameter(torch.empty(6 * dim, dtype=dtype, device=device))
        self.attn = Attention(dim, heads, dim_head, eps, joint=True, **kw)
        self.ff = FeedForward(dim, int(dim * mlp_ratio), **kw)
        self.ff_context = FeedForward(dim, int(dim * context_mlp_ratio), **kw)

    def forward(self, x, context, temb, pe, transformer_options={}):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(temb).float().unsqueeze(1).chunk(6, dim=-1)
        table = comfy.model_management.cast_to(self.scale_shift_table_context, dtype=torch.float32, device=x.device)
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = table.chunk(6, dim=-1)

        norm_x = (layer_norm(x.float(), self.eps) * (1 + scale_msa) + shift_msa).type_as(x)
        norm_c = (layer_norm(context.float(), self.eps) * (1 + c_scale_msa) + c_shift_msa).type_as(context)
        attn_x, attn_c = self.attn(norm_x, norm_c, pe, transformer_options=transformer_options)
        del norm_x, norm_c
        x = (x.float() + attn_x * gate_msa).type_as(x)
        context = (context.float() + attn_c * c_gate_msa).type_as(context)
        del attn_x, attn_c

        norm_x = (layer_norm(x.float(), self.eps) * (1 + scale_mlp) + shift_mlp).type_as(x)
        norm_c = (layer_norm(context.float(), self.eps) * (1 + c_scale_mlp) + c_shift_mlp).type_as(context)
        x = (x.float() + self.ff(norm_x) * gate_mlp).type_as(x)
        context = (context.float() + self.ff_context(norm_c) * c_gate_mlp).type_as(context)
        return x, context


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear_1 = operations.Linear(in_channels, dim, dtype=dtype, device=device)
        self.linear_2 = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, x):
        return self.linear_2(torch.nn.functional.silu(self.linear_1(x)))


class PAN2Transformer3DModel(nn.Module):
    def __init__(self, patch_size=(1, 2, 2), in_channels=97, out_channels=48, num_attention_heads=32, attention_head_dim=128,
                 num_layers=44, num_refiner_layers=4, mlp_ratio=8.0, context_mlp_ratio=4.0, refiner_mlp_ratio=4.0, text_embed_dim=4096,
                 rope_axes_dim=(16, 56, 56), rope_theta=256.0, rope_temporal_scale=16 / 24, image_model=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.patch_size = tuple(patch_size)
        self.out_channels = out_channels
        self.rope_axes_dim = tuple(rope_axes_dim)
        self.rope_theta = rope_theta
        self.rope_temporal_scale = rope_temporal_scale
        dim = num_attention_heads * attention_head_dim
        kw = {"dtype": dtype, "device": device, "operations": operations}

        self.x_embedder = operations.Conv3d(in_channels, dim, kernel_size=self.patch_size, stride=self.patch_size, dtype=dtype, device=device)
        self.time_embedder = TimestepEmbedding(256, dim, **kw)
        self.context_embedder = nn.Sequential(operations.RMSNorm(text_embed_dim, eps=1e-5, dtype=dtype, device=device),
                                              operations.Linear(text_embed_dim, dim, dtype=dtype, device=device))
        self.context_refiner = nn.ModuleList([TokenRefinerBlock(dim, num_attention_heads, attention_head_dim, refiner_mlp_ratio, eps=1e-5, **kw)
                                              for _ in range(num_refiner_layers)])
        self.context_type_embedding = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(dim, num_attention_heads, attention_head_dim, mlp_ratio, context_mlp_ratio, **kw)
                                                 for _ in range(num_layers)])
        self.norm_out = AdaLayerNorm(dim, 2, **kw)
        self.proj_out = operations.Linear(dim, math.prod(self.patch_size) * out_channels, dtype=dtype, device=device)

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, transformer_options={}, **kwargs):
        transformer_options = transformer_options.copy()
        batch, _, frames, height, width = x.shape
        p_t, p_h, p_w = self.patch_size
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
        t_len, h_len, w_len = x.shape[2] // p_t, x.shape[3] // p_h, x.shape[4] // p_w
        device, dtype = x.device, x.dtype

        pe = rope_freqs((t_len, h_len, w_len), self.rope_axes_dim, self.rope_theta, self.rope_temporal_scale, device)
        x = self.x_embedder(x).flatten(2).transpose(1, 2).contiguous()
        temb = self.time_embedder(timestep_embedding(timestep.reshape(-1)).float())  # the modulation runs in float32

        context = self.context_embedder(context.to(dtype))
        for block in self.context_refiner:
            context = block(context, transformer_options=transformer_options)
        context = context + comfy.model_management.cast_to(self.context_type_embedding, dtype=context.dtype, device=device)

        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "double"
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.transformer_blocks), device, transformer_options)
        for i, block in enumerate(self.transformer_blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype)
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(args["img"], args["txt"], args["vec"], args["pe"], transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": temb, "pe": pe, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x, context = out["img"], out["txt"]
            else:
                x, context = block(x, context, temb, pe, transformer_options=transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

        scale, shift = self.norm_out(temb).float().unsqueeze(1).chunk(2, dim=-1)
        x = self.proj_out((layer_norm(x.float()) * (1 + scale) + shift).to(dtype))
        x = x.reshape(batch, t_len, h_len, w_len, self.out_channels, p_t, p_h, p_w)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return x[:, :, :frames, :height, :width]
