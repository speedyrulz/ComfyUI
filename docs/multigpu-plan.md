# Multi-GPU mode: staged design

2026-11-14. Target machine: RTX 5060 Ti 16 GB plus RTX 4060 Ti 16 GB under WSL. The design is written so it also works for other mixed-generation NVIDIA pairs.

## Decisions

- **Usage:** mostly single interactive images and long video jobs, so per-run time comes first. The core of the mode is placement and residency (which model lives on which card). Sampling is split across cards only where the work allows it.
- **Queues:** two ComfyUI processes (one per GPU) are acceptable. There is no cross-prompt pipelining in the executor. The benchmark runner measures the two-worker setup.
- **Reference workflows:** Qwen Image 2.1 Edit at 4 MP, MiniMax H3 image t2i, MiniMax H3 t2v.
- **Cross-card results:** work may move to the 4060 Ti automatically. Because the same seed gives a different image there, every such move logs a warning and is flagged in benchmark reports.

## 1. Starting point

### Hardware (measured in earlier sessions)

| | RTX 5060 Ti | RTX 4060 Ti |
|---|---|---|
| Architecture | Blackwell, sm_120 | Ada, sm_89 |
| Link | PCIe 3.0 x8 (CPU slot) | PCIe 3.0 x4 (chipset slot) |
| Pinned RAM to GPU | 5.8-6.0 GB/s | 2.5 GB/s |
| Native formats | bf16, fp8, int8, NVFP4, MXFP8 | bf16, fp8, int8 (NVFP4 emulated, 2-3.6x slower per layer) |
| Qwen Image 2.1, 4 MP, int8 + Sage | 4.5-4.75 s/step | 5.71 s/step |
| Klein 9B upscale, 33k tokens | 6.5 s/step (NVFP4 + Sol) | 10.8 s/step (int8 + Sol) |
| MiniMax H3 int8, 4 MP | 1.86 s/step | 2.72 s/step |

- No P2P. Card-to-card copies go through RAM at about 1.9 GB/s.
- On real models the 5060 Ti is 1.2-1.46x faster at the same format, not the 1.1x a synthetic block shows. Placement therefore uses per-workflow measurements, not one fixed speed ratio.
- Torch and nvidia-smi number the cards in opposite orders. The mode names GPUs by device name, never by bare index.
- The 4060 Ti loads weights at well under half the 5060 Ti's rate. It is a good home for models that load once and stay resident, and a poor one for models that are swapped in every run.
- The 4060 Ti's chipset link probably shares the chipset uplink with any chipset-attached NVMe. Check which M.2 slot holds the models drive.

### Your reference workflows

| Workflow | Prompt enhancer | Text encoder | Diffusion | Notes |
|---|---|---|---|---|
| Qwen Image 2.1 Edit, 4 MP | Qwen Image 2.1 PE, w4a8, 6.1 GB, through WAS Generate Text (the Qwen to Klein variant uses llama-server instead) | `qwen3vl_8b_nvfp4`, 7.5 GB, fast only on the 5060 Ti | Qwen Image 2.1 int8 with NVFP4 MLP layers, 5.8 GB (NVFP4 layers native only on the 5060 Ti) | 8 steps, CFG 1 |
| H3 image t2i | Same PE through Generate Text | `qwen3vl_32b_minimax_h3_nvfp4_awq`, 15.7 GB, fast only on the 5060 Ti | `minimax_h3_image_pruned_int8_convrot`, 21 GB, partly streamed | 6 steps, CFG 1 |
| H3 t2v | Same PE through WAS Generate Text | Same NVFP4 TE | `minimax_h3_fl2va_pruned_int8_convrot`, 21 GB, partly streamed | 73 frames at 1344x768, 4 steps, plus audio VAE |

On one 16 GB card the stages evict each other (PE → TE → DiT → VAE). The two workflows also pose different problems:
- **Qwen:** everything is small. PE plus TE (13.6 GB) could stay resident on the 4060 Ti, and the DiT plus VAE on the 5060 Ti, so nothing would be swapped. The cost is running the NVFP4 TE emulated on the 4060 Ti for one encode per run.
- **H3:** the TE alone nearly fills a card and the DiT does not fit at all, so some model is reloaded on every run. The question is which one, and over which link. In earlier H3 runs the sampler was about 15% of a 60-100 s run, a prompt enhancer with thinking on took about 60 s of a 92 s run, and H3 "sampling including load" was 21-23 s against about 11 s of step compute. The biggest wins are fewer swaps and the right card for each model. Splitting one image's sampling comes second.

### What the fork already has

- `--default-device` and `--cuda-device` control which GPUs are visible and which is the default.
- Select Model / CLIP / VAE Device nodes (`comfy_extras/nodes_multigpu.py`) move a component to `gpu:N`. They reload it there through the loader's `cached_patcher_init` and pick a supported compute dtype per device.
- MultiGPU CFG Split clones the model onto the other GPU and splits **conds** across devices with one thread per GPU (`comfy/multigpu.py`, `_calc_cond_batch_multigpu` in `comfy/samplers.py`). At CFG 1 there is one cond, so it gives nothing.
- `load_balance_devices()` and the MultiGPU Options node (relative speed) exist but are unused: the node is unregistered and the scheduler ignores speed.
- Dynamic VRAM (comfy-aimdo) already initializes every visible device.
- Loaders already accept `model_options["load_device"]` (`comfy/sd.py`), so placement can be chosen at load time without a reload.
- Generate Text (`TextGenerateV2`) has a `system_prompt` input, and Preview as Text returns text through the API, so the LLM advisor can run on core nodes.
- `evaluate_context_windows()` in `comfy/context_windows.py` already takes unused `device` and `first_device` parameters, a sign that someone upstream planned to run context windows on multiple GPUs.

Missing: batch and context-window splitting, automatic placement, thread safety in model management (`current_loaded_models` has no lock), and any device awareness in the executor.

## 2. Feasibility verdicts

| Idea | Verdict | Reason |
|---|---|---|
| Placement: PE, TE, VAE on the 4060 Ti, diffusion model on the 5060 Ti | **Yes, Stage 2** | Stops swaps on the 5060 Ti. The Select nodes can already do it per workflow, but there is no mode or plan. |
| Second GPU instead of a swap | **Yes, Stage 3** | Depends on the workload (worked examples in Stage 3), so it needs a cost model. |
| Batch split (batch ≥ 2) | **Yes, Stage 4** | Speedup is about 2/r, where r is the 4060 Ti/5060 Ti step-time ratio: Qwen ~1.6x, H3 ~1.37x. A second candidate image costs about 25-45% more time instead of 100%. |
| Context-window split (long video) | **Yes, Stage 4** | Windows are independent within a step and currently run one after another. Only applies to videos long enough to use Context Windows; your 73-frame H3 video runs as a single window. |
| CFG split | **Exists** | For CFG > 1 workflows only. |
| Tiled VAE decode across both GPUs | **Maybe, Stage 5** | Worth it only if video decode is a large share of a run. |
| Tiled low-denoise refine/upscale | **Maybe, Stage 5** | Full-generation tiling duplicated subjects (2026-11-05). At low denoise the layout is already fixed. |
| Model prefetch while the other card runs PE/TE | **Maybe, Stage 5** | Helps cold loads only. Once Stage 2 keeps models resident, there is little left to hide. |
| Two workers for queues | **Yes, no code** | Measured 1.80x on Krea 2 when each worker takes the next job when it is free. |
| Parallel branches within one prompt ("lanes") | **Not planned** | Your workflows are single chains. It is a large executor change that upstream merges would keep conflicting with. |
| Cross-prompt pipelining | **Not planned** | Queues are rare and two workers cover them. |
| Layer (pipeline) split for models > 16 GB | **Marginal** | No speedup at batch 1. It only removes weight streaming, which already mostly overlaps compute on the 5060 Ti, and half the layers would run on the slower card. |
| Tensor or sequence parallel; DistriFusion patch parallel | **No** | Each block moves about 100 MB across a 1.9 GB/s link. 60 blocks ≈ 3 s of transfer per step, against the ~2.3 s per card that splitting would save. |
| Step-parallel sampling (ParaDiGMS) | **No** | Spends 2-3x the compute to save time with many GPUs. Two GPUs and 4-8 step schedules gain nothing. |
| Using 4060 Ti VRAM as weight storage for the 5060 Ti | **No** | 4060 Ti → 5060 Ti runs at 1.9 GB/s through RAM, while pinned RAM → 5060 Ti runs at 5.8 GB/s. |

## 3. Design rules

- Opt-in with `--multigpu`. With it off, behavior matches upstream.
- GPUs are named by device name (unique substring allowed) in plan files, logs and reports.
- Capability first: a model goes only to devices that run its weight formats natively (for example NVFP4 only on sm_120), unless the plan forces otherwise.
- Keep models resident and move the small data (latents, conditioning), not the weights.
- Running a diffusion model, or any batch items, on a card other than the primary logs one warning per model per session, for example: `Sampling MiniMaxH3 on RTX 4060 Ti: the same seed gives a different image than on RTX 5060 Ti.`
- Most new code lives in `comfy/multigpu.py` and `comfy_extras/nodes_multigpu.py`, with small hooks in shared files, so upstream merges stay cheap. `execution.py` is not touched.
- Every stage can be switched off and is measured with the Stage 1 runner before and after.
- No network code. The runner talks only to the local server, and an outside LLM is used through an exported file.

## 4. Stages

### Stage 1: Benchmark runner and advisor export (no core changes)

**Goal:** measure the three reference workflows under different placements, using the nodes that already exist, before building anything else.

- `script_examples/multigpu_benchmark.py <workflow>`, where the workflow is an API-format `.json` or a `.png`/video saved by ComfyUI (they carry the prompt). Uses aiohttp, which is already a dependency, against `127.0.0.1`. The server must see both GPUs: launch it with `--default-device 1`, not `--cuda-device 1`. `--list` shows the GPUs, loaders and variants without running anything.
- **Variants** are made by rewriting the API graph: Select Model / CLIP / VAE Device is inserted after each loader (any node that loads a model file and outputs a MODEL, CLIP or VAE), including loaders inside subgraphs. The default sweep:
  - baseline: everything on the primary, as today;
  - each text encoder on its own on the other GPU (the PE loads through CLIPLoader, so it counts as one);
  - all text encoders on the other GPU;
  - all VAEs on the other GPU;
  - text encoders and VAEs on the other GPU;
  - diffusion model on the other GPU, to measure r for this workflow.

  `--variant "clip:*PE*=gpu:1,vae:*=gpu:1"` (repeatable, glob patterns over loader names) runs explicit placements instead. The llama-server enhancer is an outside process and cannot be moved per run.
- **Runs:** `/free` (unload models and clear the cache), 1 cold run, then N warm runs. Run k sets every seed input to the workflow's seed + k, so whatever depends on a seed re-runs while loaders stay cached, and run k of every variant is directly comparable. In the H3 workflows the enhancer's seed is linked to the seed node, so warm runs cover PE → TE → sampler → VAE. `--set NODE.INPUT=VALUE` changes inputs first, for example fewer steps for a long video job; placement costs do not depend on the step count.
- **Measurements:** start and end of each node from `progress_state` websocket events; prompt total; peak VRAM per GPU from polling `/system_stats`; output file paths for side-by-side comparison. Variants that sample on the 4060 Ti carry the cross-card note.
- **Two-worker mode:** `--servers 127.0.0.1:8191,127.0.0.1:8192 --jobs K` runs K jobs on the first server alone, then K jobs across all servers, each server taking the next job when it is free. Start one single-GPU instance per card (`--cuda-device 1` is the 5060 Ti, `--cuda-device 0` the 4060 Ti) with separate output directories. It reports wall time, the median job time per server, and the long-queue rate from those medians.
- **Output:** `output/multigpu_bench/<name>-<time>/` with `report.json`, `report.md` (ranked table with speedup over baseline) and `advisor_prompt.md` (system prompt, hardware profile, results and plan schema, ready to paste into any LLM). The reports are rewritten after every variant, so a stopped sweep still leaves one. `--notes` adds hardware facts the server cannot report, such as PCIe links.
- **Local LLM:** `--advisor <clip file> --advisor-type <type>` sends a small graph to the server (CLIPLoader → Generate Text with `system_prompt` → Preview as Text) and saves the reply. No new nodes are needed. Passing a report directory instead of a workflow asks again without re-running the benchmark.
- The runner ranks variants itself. The LLM is optional: it explains trade-offs and suggests combinations that were not measured.

**Test:** baseline all three workflows. Warm-run spread should be below 5%. Check that the GPU names in the report match the actual cards.
**Size:** one new script and no changes to core files.

**Status:** built and tested, except that the advisor's no-speculation rule is untested. Not yet run: Qwen Image 2.1 Edit and H3 image t2i (no saved output with an embedded prompt was found), and the H3 sweep with `--runs 3`.

#### Results

- **Krea 2 T2I + 2x iterative upscale, 1 warm run (2026-11-14):** baseline 84.3 s warm, 161 s cold. Text encoder and VAE on the 4060 Ti: 82.5 s warm, 140 s cold. The 4060 Ti peaked at 3.7 GiB, which confirms the move.
- **H3 t2v at 8 steps (`--set 160.value=8`), 1 warm run (2026-11-14):** baseline 261 s warm (sampler 223, PE 13, TE encode 9, VAE decode 8). TE on the 4060 Ti: encode 9 → 22 s, sampler −6 s, a net loss. VAE on the 4060 Ti: decode 7.9 → 10.3 s, a net loss. Diffusion model on the 4060 Ti: sampler 307 s (r ≈ 1.38). PE on the 4060 Ti fails (below). Today the best H3 placement is everything on the 5060 Ti.
- **Noise:** PE time varied 13-48 s on the same GPU because MTP "auto" makes the output length vary. Use `--runs 3` and compare the node table.
- **Two workers, Krea 2 (2026-11-16):** 5 jobs on the 5060 Ti alone took 395 s (median 78 s per job). 5 jobs across both servers, each taking the next job when free, took 220 s: **1.80x**, or about 1.89x for a long queue. While both ran, the median job took 72 s on the 5060 Ti and 97 s on the 4060 Ti; an earlier run measured 127 s on the 4060 Ti, so r on this workflow is somewhere between 1.3 and 1.6. Round-robin dispatch of 4 jobs gave only 1.33x, because the slower card set the wall time.

#### Blocker for Stage 2: text generation on the second GPU

Text generation (Qwen3.5 PE through core `TextGenerateV2` or WAS Fast Generate Text) on the non-default GPU through Select CLIP Device either crashes with `CUDA error: invalid argument` (`Unable to access operand ... in the specified context`) in `cuMemcpyHtoDAsync`, or stalls at 60-90 s per token with both GPUs idle. The crash path is `qwen35.py _generate_mtp` → `model_prefetch.prefetch_queue_pop` → `pin_modules` → `ops.cast_modules_with_vbar` → `model_management.cast_to_gathered`.

- Happens with MTP on and off. H3 TE encoding on the second GPU works.
- The same model generates normally (24.6 s cold) when the 4060 Ti is the only visible GPU.
- Pins are registered portable, `CLIP.generate` already runs in `cuda_device_context`, and VBARs and pin state are kept per load device (`dynamic_vbars`, `dynamic_pins`).
- Select CLIP Device builds the reloaded model for `cuda:0` through `cached_patcher_init`, then retargets it to `cuda:1`. Loading directly on the target device (`model_options["load_device"]`), as Stage 2 plans, may avoid the bug.

Stage 2 starts by reproducing this with a standalone load on `cuda:1`.

### Stage 2: `--multigpu` and a placement plan

**Goal:** both GPUs always available, with every component placed by a plan instead of per-workflow Select nodes.

- `--multigpu` in `comfy/cli_args.py`. It needs both GPUs visible (`--default-device 1`).
- Plan file at `user/default/multigpu.json` (format in §5). `main.py` reads it and passes a plain dict to `comfy.multigpu`, so the app-level file handling stays out of `comfy/`.
- Default plan when no file exists: diffusion → primary, text encoders (which includes the PE, since it loads through CLIPLoader) and CLIP vision → aux, VAE → primary. Stage 1 results replace this default.
- Hooks: in multigpu mode, the default `load_device` in the `comfy/sd.py` loaders comes from `comfy.multigpu` by role and file name. `text_encoder_device()` and `vae_device()` follow the plan. Offload stays on CPU. Select nodes still override per workflow.
- The cross-card warning from §3 is added here.
- Device combo tooltips show the mapping (`gpu:0 = RTX 5060 Ti, gpu:1 = RTX 4060 Ti`). The saved values stay `gpu:N`, so existing workflows keep working.
- llama-server (Qwen to Klein workflow): in this mode it inherits both GPUs and by default spreads layers across them, adding card-to-card traffic for every token. Pin it to one card: llama.cpp reads `LLAMA_ARG_*` env vars for most flags (to verify for `--device`), or add a one-line device option to the third-party enhancer node.

**Test:** all three workflows against the Stage 1 baseline. The log should show no PE or TE load on the 5060 Ti, and the diffusion model's resident part should not be evicted between runs. A TE on the other card can change embeddings slightly, so check images side by side.
**Size:** small. Touches `cli_args.py`, `main.py`, `comfy/multigpu.py`, a few lines in `sd.py` and `model_management.py`, and the node tooltips.

### Stage 3: Capability-aware automatic placement and residency

**Goal:** choose devices automatically, including "use the 4060 Ti instead of swapping on the 5060 Ti" and the reverse.

- **Device profile at startup:** name, UUID, compute capability, VRAM, native formats (from the existing `supports_*_compute` functions), and measured pinned upload bandwidth (about 0.2 s to measure).
- **Model descriptor at load time:** role, file, size, and weight formats from quant metadata.
- **Step-time history** per (model, device, resolution bucket), imported from benchmark reports and updated after each sampler run. The app layer saves it as JSON in the user directory.
- **Decision** among devices with native support: `cost = time to load the non-resident bytes over that card's link + estimated compute + time to reload anything it would evict that was used in the last run`. Pick the lowest cost and log one short line with the numbers.
- **Worked examples:**
  - Qwen Edit → Klein upscale, 4 Klein steps: on the 5060 Ti (NVFP4 + Sol) 26 s, on the 4060 Ti (int8 + Sol) 43 s. Even if the 5060 Ti has to swap Qwen and Klein, a swap of these 6-8 GB files costs about 1-1.5 s each way, so the 5060 Ti wins by over 10 s.
  - H3 TE (NVFP4, 15.7 GB): on the 4060 Ti it runs emulated, and it cannot stay resident there alongside the PE, so one of them crosses the x4 link every run (15.7 GB ≈ 6 s). On the 5060 Ti it pushes out part of the DiT, which streams back at about 5.8 GB/s (≈ 2.7 s). The likely answer is TE on the 5060 Ti and PE on the 4060 Ti, but Stage 1 numbers decide.
- The decision is made at load time. A model is moved to the other card later only when the estimated saving exceeds the cost of reloading it there.

**Test:** the three workflows plus Qwen Edit → Klein, and a session that alternates H3 and Qwen runs, against the Stage 2 static plan.
**Size:** medium, mostly in `comfy/multigpu.py`, called from the Stage 2 hooks.

### Stage 4: Splitting sampling where the work allows it

Both parts reuse the existing per-GPU threads and model clones from MultiGPU CFG Split. The existing node keeps its id and gets the display name "MultiGPU Work Split". Under `--multigpu`, Stage 3 applies it automatically when a second copy fits.

**4a. Batch split (batch ≥ 2).** In `_calc_cond_batch_multigpu`, when there are fewer work units than devices and the cond batch is > 1, split along dim 0: `input_x`, `mult`, and the batch-aligned conditioning tensors. Add a batch-narrowing helper to `comfy/conds.py` for the cond types that repeat to batch size. List-type conds such as reference latents need checking per model. Weight the split with the existing `load_balance_devices()`, using speeds from the plan or benchmark history. Expected at batch 2: Qwen ~1.6x (int8 file), H3 ~1.37x. Your current Qwen file has NVFP4 MLP layers that the 4060 Ti emulates, so for that file Stage 3 decides whether a split still pays.

**4b. Context-window split (long video).** `IndexListContextHandler.execute()` evaluates windows one by one. Hand each GPU a share of the windows per step, weighted by speed, with its own copy of `transformer_options` (it currently writes `context_window` into a shared dict). Then combine the results on the main thread in window order, so output does not depend on thread timing. Expected ≈ 2/r for an even window count, about 1.37x at H3's ratio. Check upstream first, given the unused `device` parameters.

**Test:** Qwen and H3 at batch 2 and 4; a long H3 video with Context Windows. Work done on the primary must be bit-identical to single-GPU output. Also check VRAM and RAM with two model copies, interrupt during a split, and the patch nodes (Spectrum, BlockSparseAttention, Qwen cache, RulZ sparse) on the cloned model.
**Size:** medium, in `samplers.py`, `conds.py`, `context_windows.py` and `multigpu.py`.

### Stage 5: Experiments, each gated by Stage 1-4 data

- **Tiled VAE decode across both GPUs** for video: run tiles from the existing tiled decode on both cards with a VAE copy on each. Build only if decode is more than about 10% of an H3 t2v run.
- **Model prefetch:** warm the diffusion model on the 5060 Ti while PE/TE run on the 4060 Ti. Needs a model-management lock, and a check that aimdo can fill VRAM outside a forward pass. Build only if cold-load time is still more than about 10% of a run after Stage 3.
- **Tiled low-denoise refine/upscale:** a helper node that builds overlapping, feathered area conds for the existing scheduler, for a Klein upscale at denoise 0.2-0.4. Use ConditioningSetArea, not ...Percentage, which crashes on 3D latents.
- **Layer split** for models above 16 GB: only if streaming stalls take more than about 20% of a step on the 5060 Ti.

### Checks for every stage

- Runner baseline before the change and comparison after, using the stage's on/off switch.
- Paths that stay on the same card must give bit-identical images with fixed seeds. Paths that move to the other card get a side-by-side check and the warning.
- Unit tests in `tests-unit/` for the pure logic: plan parsing, the placement cost model, batch and window slicing.
- WSL: watch for more frequent transient aimdo `cuMemSetAccess` errors with two devices.

## 5. Plan file

Illustrative only; Stage 1 results produce the real one.

```json
{
  "primary": "RTX 5060 Ti",
  "roles": {
    "diffusion": "primary",
    "text_encoder": "RTX 4060 Ti",
    "clip_vision": "RTX 4060 Ti",
    "vae": "primary"
  },
  "models": {
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": {"device": "RTX 5060 Ti"},
    "minimax_h3_image_pruned_int8_convrot.safetensors": {"relative_speed": {"RTX 5060 Ti": 1.0, "RTX 4060 Ti": 0.68}}
  },
  "batch_split": true
}
```

The runner reports in this format and the advisor is asked to answer in it, so a recommendation can be saved directly as the plan.

## 6. Advisor system prompt (draft)

> You configure a ComfyUI multi-GPU placement plan. You get a hardware profile (GPU names, compute capability, VRAM, native weight formats, host link bandwidth, whether P2P exists) and benchmark results: per-variant total time, per-node times, peak VRAM per GPU, cold and warm runs. Rules: prefer measured numbers to assumptions; never place a model on a GPU without native support for its weight format unless the results show it is still faster; a model that does not fit in VRAM is streamed over the host link each step, so link bandwidth matters for it; a GPU with a slow link is a good home only for models that stay resident; card-to-card copies go through system RAM; the same seed gives different images on different GPUs, so say when a choice moves sampling to another card. Output (1) a JSON plan in the given schema, (2) for each choice, the measured numbers behind it, (3) at most three extra variants worth benchmarking, with the expected gain. Do not recommend splitting one image's sampling across GPUs except through batch or context-window splitting.
