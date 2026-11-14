<div align="center">

# ComfyUI — speedyrulz fork

**A fork of [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) with PAN 2 video support and a few speed, memory and bug fixes.**

[Wiki](https://github.com/speedyrulz/ComfyUI/wiki) · [Fork changes](https://github.com/speedyrulz/ComfyUI/wiki/Fork-Changes) · [Installation](https://github.com/speedyrulz/ComfyUI/wiki/Installation) · [Upstream](https://github.com/Comfy-Org/ComfyUI) · [Docs](https://docs.comfy.org)

<img width="1590" height="795" alt="ComfyUI Screenshot" src="https://github.com/user-attachments/assets/36e065e0-bfae-4456-8c7f-8369d5ea48a2" />
</div>

ComfyUI is a node-graph engine for generating images, video, 3D, audio and text with open models, running locally on Windows, Linux and macOS. It supports the latest open models natively and streams weights between RAM and VRAM, so even very large models run on consumer GPUs.

This fork follows upstream `master` and is merged with it regularly, so it has every upstream model, node and fix. The differences are listed below; if you don't use them, it behaves exactly like upstream.

## What's different in this fork

| Area | Change | Details |
|---|---|---|
| **PAN 2** | Text-to-video and image-to-video with IFM's [PAN 2](https://huggingface.co/IFM/PAN2) (about 29B parameters), through the standard loaders and samplers, plus nodes for prompts that stay as written, the released negative prompt, and first-frame conditioning | [PAN 2](https://github.com/speedyrulz/ComfyUI/wiki/PAN-2) |
| **PAN 2 speed** | Spectrum step forecasting (sampling in 58 s instead of 149 s at 50 steps in testing), a memory-efficient Sage attention node (25.7 s instead of 42.6 s per step), token-sliced float32 math, and a fused RMSNorm + RoPE kernel | [PAN 2 Speed and Memory](https://github.com/speedyrulz/ComfyUI/wiki/PAN-2-Speed-and-Memory) |
| **Qwen-Image 2.1** | Reference images are encoded once and shared by the positive and negative prompts, instead of twice | [Prompt Encoding](https://github.com/speedyrulz/ComfyUI/wiki/Qwen-Image-Prompt-Encoding) |
| **WSL2 and multi-GPU** | The Windows pinned-memory budget under WSL2, and locking so two GPUs can stream weights at the same time | [Pinned Memory and Multi-GPU](https://github.com/speedyrulz/ComfyUI/wiki/Pinned-Memory-and-Multi-GPU) |
| **Loading** | Qwen3.5 text encoders in the transformers layout, and Wan 2.2 VAEs in the diffusers layout | [Fork Changes](https://github.com/speedyrulz/ComfyUI/wiki/Fork-Changes#loading) |
| **Fixes** | OFT and BOFT LoRAs on bf16/fp16 models, saving int8-quantized models, the Llama causal mask with a cached prefix | [Fork Changes](https://github.com/speedyrulz/ComfyUI/wiki/Fork-Changes#other-fixes) |

Every change, with its reason and commit, is on [Fork Changes](https://github.com/speedyrulz/ComfyUI/wiki/Fork-Changes).

## Quick start

The desktop app, the Windows portable builds and comfy-cli install upstream ComfyUI. To run this fork, install it from git. For NVIDIA:

```bash
git clone https://github.com/speedyrulz/ComfyUI
cd ComfyUI
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python main.py
```

Then open http://127.0.0.1:8188. For AMD, Intel, Apple silicon and other hardware, model folders, and ComfyUI-Manager, see [Installation](https://github.com/speedyrulz/ComfyUI/wiki/Installation).

## Features

Everything upstream ComfyUI offers, including:

- A visual node graph for image, video, audio, 3D and text workflows, with reusable subgraphs, templates, App Mode and a local API.
- Native support for current open models: Flux.2, Qwen Image, Wan 2.2, LTX-Video 2, HunyuanVideo 1.5, MiniMax H3, ACE-Step and many more, plus PAN 2 in this fork ([full list](https://github.com/speedyrulz/ComfyUI/wiki/Using-ComfyUI#what-comfyui-supports)).
- Efficient local execution: only changed parts of a graph re-run, smart VRAM and RAM management, weight streaming, and quantized models.
- Runs fully offline: the core downloads nothing unless you ask it to.
- Custom nodes, extra model paths, and high bit depth and HDR image and video formats.

## Documentation

| Wiki page | Contents |
|---|---|
| [Installation](https://github.com/speedyrulz/ComfyUI/wiki/Installation) | Every OS and GPU, model folders, ComfyUI-Manager, optional extras |
| [Running and Configuration](https://github.com/speedyrulz/ComfyUI/wiki/Running-and-Configuration) | Command-line options, previews, TLS, frontend versions, model paths |
| [Using ComfyUI](https://github.com/speedyrulz/ComfyUI/wiki/Using-ComfyUI) | How graphs run, prompt weighting, dynamic prompts, embeddings |
| [Keyboard Shortcuts](https://github.com/speedyrulz/ComfyUI/wiki/Keyboard-Shortcuts) | Queue, workflow, node and canvas shortcuts |
| [Troubleshooting](https://github.com/speedyrulz/ComfyUI/wiki/Troubleshooting) | Common errors and fixes |
| [Maintaining the Fork](https://github.com/speedyrulz/ComfyUI/wiki/Maintaining-the-Fork) | Syncing with upstream, making changes, tests |
| [Upstream Project](https://github.com/speedyrulz/ComfyUI/wiki/Upstream-Project) | Release process, frontend development, support channels |

Upstream's documentation is at [docs.comfy.org](https://docs.comfy.org), with workflow templates at [comfy.org/workflows](https://comfy.org/workflows).

## Support

This fork has no issue tracker. For anything that also happens on upstream ComfyUI, use upstream's [issues](https://github.com/Comfy-Org/ComfyUI/issues), [Discord](https://comfy.org/discord) or [Matrix](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org). Upstream doesn't have the fork's additions, so please don't report problems with them there.

## License

[GPL-3.0](LICENSE), the same as upstream ComfyUI.
