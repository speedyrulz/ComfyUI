"""Benchmark a workflow with its models placed on different GPUs.

The workflow is an API-format .json (Export (API) in the workflow menu), or a .png
or video that ComfyUI saved, which carries the prompt in its metadata. Variants are
made by inserting Select Model/CLIP/VAE Device nodes after the loaders, so the server
must see every GPU: start it with --default-device N, not --cuda-device N.

Only the given local servers are contacted.

    python script_examples/multigpu_benchmark.py output/ComfyUI_00001_.png --list
    python script_examples/multigpu_benchmark.py workflow_api.json --runs 3
    python script_examples/multigpu_benchmark.py workflow_api.json --variant "clip:*PE*=gpu:1,vae:*=gpu:1"
    python script_examples/multigpu_benchmark.py workflow_api.json --advisor qwen3_8b.safetensors --advisor-type qwen_image
    python script_examples/multigpu_benchmark.py output/multigpu_bench/workflow_api-20261114-190000 --advisor qwen3_8b.safetensors --advisor-type qwen_image
    python script_examples/multigpu_benchmark.py workflow_api.json --servers 127.0.0.1:8188,127.0.0.1:8189 --jobs 6
"""

import argparse
import asyncio
import copy
import datetime
import fnmatch
import json
import re
import statistics
import time
import uuid
from pathlib import Path

import aiohttp
import av
from PIL import Image

MODEL_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf")
SELECT_NODES = {"MODEL": ("SelectModelDevice", "model"), "CLIP": ("SelectCLIPDevice", "clip"), "VAE": ("SelectVAEDevice", "vae")}
SEED_INPUTS = ("seed", "noise_seed")
GIB = 1024 ** 3

SYSTEM_PROMPT = """You configure where a ComfyUI workflow's models run on a multi-GPU machine. You get the GPUs, the workflow's model loaders, and benchmark results: per-variant warm and cold run times, the slowest nodes per variant, and peak VRAM per GPU. Peak VRAM counts every process on the GPU, including the desktop.

Rules:
- Prefer measured numbers to assumptions, and say when a choice rests on an assumption.
- A model whose weight format a GPU does not support natively (for example NVFP4 below Blackwell) runs emulated and much slower there.
- A model that does not fit in VRAM is streamed over the host link every step, so link bandwidth matters for it. A GPU on a slow link is a good home only for models that stay loaded.
- Copies between GPUs go through system RAM unless the notes say P2P works.
- The same seed gives a different image on a different GPU, so say when a choice moves sampling to another GPU.
- A failed variant means that placement does not work in this ComfyUI build. Do not guess why, and do not recommend it.

Answer with:
1. A JSON plan in the given schema.
2. For each choice, the measured numbers behind it.
3. At most three more variants worth benchmarking, with the gain you expect."""

PLAN_SCHEMA = """{
  "primary": "<GPU name>",
  "roles": {"diffusion": "primary", "text_encoder": "<GPU name>", "clip_vision": "<GPU name>", "vae": "primary"},
  "models": {"<model file>": {"device": "<GPU name>"}},
  "batch_split": false
}"""


def load_workflow(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        graph = json.loads(path.read_text(encoding="utf-8"))
        if "nodes" in graph and "links" in graph:
            raise SystemExit(f"{path} is a UI workflow. Use Export (API) in the workflow menu, or pass an image or video saved by it.")
        return graph
    if path.suffix.lower() == ".png":
        prompt = Image.open(path).info.get("prompt")
    else:
        with av.open(str(path)) as container:
            prompt = container.metadata.get("prompt")
    if prompt is None:
        raise SystemExit(f"{path} has no embedded prompt.")
    return json.loads(prompt)


def set_inputs(graph, items):
    for item in items:
        target, _, value = item.partition("=")
        node_id, _, name = target.partition(".")
        if node_id not in graph or not name:
            raise SystemExit(f"--set '{item}': expected NODE_ID.INPUT=VALUE with a node id from the workflow.")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        graph[node_id]["inputs"][name] = value


def find_loaders(graph, object_info):
    """Nodes that create a MODEL, CLIP or VAE from a model file without taking one as input."""
    loaders = []
    for node_id, node in graph.items():
        info = object_info.get(node["class_type"])
        model_file = next((v for v in node["inputs"].values() if isinstance(v, str) and v.lower().endswith(MODEL_EXTENSIONS)), None)
        if info is None or model_file is None:
            continue
        input_types = [v[0] for section in ("required", "optional") for v in info["input"].get(section, {}).values()]
        if any(t in SELECT_NODES for t in input_types if isinstance(t, str)):
            continue
        for index, output_type in enumerate(info["output"]):
            if output_type in SELECT_NODES:
                loaders.append({"id": node_id, "index": index, "type": output_type, "class_type": node["class_type"],
                                "label": f"{SELECT_NODES[output_type][1]}:{Path(model_file).name}"})
    labels = [l["label"] for l in loaders]
    for l in loaders:
        if labels.count(l["label"]) > 1:
            l["label"] += f"#{l['id']}"
    return loaders


def apply_placement(graph, loaders, placement):
    graph = copy.deepcopy(graph)
    for loader in loaders:
        device = placement.get(loader["label"])
        if device is None:
            continue
        class_type, input_name = SELECT_NODES[loader["type"]]
        select_id = f"mgpu_{loader['id']}_{loader['index']}"
        for node in graph.values():
            for name, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) == loader["id"] and value[1] == loader["index"]:
                    node["inputs"][name] = [select_id, 0]
        graph[select_id] = {"class_type": class_type, "inputs": {input_name: [loader["id"], loader["index"]], "device": device}}
    return graph


def seed_inputs(graph):
    # dynamic combo inputs are named like "sampling_mode.seed"
    return [(node_id, name) for node_id, node in graph.items() for name, value in node["inputs"].items()
            if name.split(".")[-1] in SEED_INPUTS and isinstance(value, int) and not isinstance(value, bool)]


def with_seed(graph, keys, seed):
    graph = copy.deepcopy(graph)
    for node_id, name in keys:
        graph[node_id]["inputs"][name] = seed
    return graph


def short_name(label):
    return Path(label.split(":", 1)[1]).stem


def auto_variants(loaders, gpus):
    labels = {t: [l["label"] for l in loaders if l["type"] == t] for t in SELECT_NODES}
    clips, vaes, models = labels["CLIP"], labels["VAE"], labels["MODEL"]
    variants = []
    for gpu in gpus[1:]:
        option = f"gpu:{gpu['index']}"
        groups = [(short_name(label), [label]) for label in clips]
        if len(clips) > 1:
            groups.append(("all text encoders", clips))
        if vaes:
            groups.append(("VAE", vaes))
        if clips and vaes:
            groups.append(("text encoders + VAE", clips + vaes))
        if models:
            groups.append(("diffusion model", models))
        variants += [(f"{name} on {option}", {label: option for label in group}) for name, group in groups]
    return variants


def parse_variant(spec, loaders):
    placement = {}
    for part in spec.split(","):
        pattern, _, device = part.rpartition("=")
        device = device.strip()
        matched = [l["label"] for l in loaders if fnmatch.fnmatch(l["label"], pattern.strip())]
        if not matched:
            raise SystemExit(f"--variant '{part}' matches no loader. Use --list to see loader names.")
        if device not in ("default", "cpu") and re.fullmatch(r"gpu:\d+", device) is None:
            raise SystemExit(f"--variant '{part}': the device must be default, cpu or gpu:N.")
        placement.update({label: device for label in matched})
    return spec, placement


def gpu_name(device):
    # "cuda:1 NVIDIA GeForce RTX 4060 Ti : cudaMallocAsync" -> "NVIDIA GeForce RTX 4060 Ti"
    return device["name"].split(" : ")[0].split(" ", 1)[-1]


def output_files(outputs):
    return [f"{item.get('subfolder', '')}/{item['filename']}".lstrip("/") for out in outputs.values() for items in out.values()
            if isinstance(items, list) for item in items if isinstance(item, dict) and "filename" in item and item.get("type") == "output"]


async def run_ok(server, graph):
    run = await server.run(graph)
    if run["error"]:
        raise SystemExit(f"{server.address}: {run['error']}")
    return run


class Server:
    def __init__(self, session, address):
        self.session = session
        self.address = address
        self.client_id = uuid.uuid4().hex
        self.messages = asyncio.Queue()

    async def connect(self):
        self.ws = await self.session.ws_connect(f"ws://{self.address}/ws?clientId={self.client_id}", max_msg_size=0)
        self.reader = asyncio.create_task(self.read_messages())

    async def read_messages(self):
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self.messages.put_nowait(json.loads(msg.data))

    async def get(self, path):
        async with self.session.get(f"http://{self.address}{path}") as r:
            r.raise_for_status()
            return await r.json()

    async def post(self, path, data):
        async with self.session.post(f"http://{self.address}{path}", json=data) as r:
            if r.status != 200:
                raise SystemExit(f"{self.address}{path} failed: {await r.text()}")
            if r.content_type == "application/json":
                return await r.json()

    async def gpus(self):
        return [d for d in (await self.get("/system_stats"))["devices"] if d["type"] != "cpu"]

    async def free(self, idle=None):
        """Unload models and clear the cache, then return VRAM in use per GPU once it settles.

        With idle (VRAM in use per GPU after an earlier free), also wait until every GPU is back near it.
        """
        await self.post("/free", {"unload_models": True, "free_memory": True})
        used = {}
        for i in range(60):
            await asyncio.sleep(0.5)
            now = {d["index"]: d["vram_total"] - d["vram_free"] for d in await self.gpus()}
            settled = i >= 2 and all(abs(now[k] - used.get(k, 0)) < 64 * 1024 ** 2 for k in now)
            if settled and (idle is None or all(now[k] < idle.get(k, 0) + GIB / 2 for k in now)):
                break
            used = now
        return now

    async def poll_vram(self, peak):
        while True:
            try:
                for d in await self.gpus():
                    peak[d["index"]] = max(peak.get(d["index"], 0), d["vram_total"] - d["vram_free"])
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.25)

    async def run(self, graph, timeout=None):
        prompt_id = (await self.post("/prompt", {"prompt": graph, "client_id": self.client_id}))["prompt_id"]
        result = {"nodes": {}, "cached": [], "error": None, "vram_peak": {}}
        poller = asyncio.create_task(self.poll_vram(result["vram_peak"]))
        started = None
        deadline = None if timeout is None else time.perf_counter() + timeout
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(self.messages.get(), None if deadline is None else max(deadline - time.perf_counter(), 0.1))
                except asyncio.TimeoutError:
                    result["error"] = f"timed out after {timeout:.0f} s"
                    deadline = None
                    await self.post("/interrupt", {"prompt_id": prompt_id})
                    continue
                data = msg.get("data") or {}
                if data.get("prompt_id") != prompt_id:
                    continue
                now = time.perf_counter()
                if msg["type"] == "execution_start":
                    started = now
                elif msg["type"] == "execution_cached":
                    result["cached"] = data["nodes"]
                elif msg["type"] == "progress_state":
                    for node_id, state in data["nodes"].items():
                        node = result["nodes"].setdefault(node_id, {"class_type": graph.get(node_id, {}).get("class_type", "?")})
                        if state["state"] == "running":
                            node.setdefault("start", now)
                        elif state["state"] == "finished":
                            node.setdefault("end", now)
                elif msg["type"] == "execution_error":
                    message = (data.get("exception_message") or "").strip().split("\n")[0]
                    result["error"] = f"{data.get('node_type')}: {message}"
                    break
                elif msg["type"] == "execution_interrupted":
                    result["error"] = result["error"] or "interrupted"
                    break
                elif msg["type"] == "execution_success":
                    break
        finally:
            poller.cancel()
        result["total"] = now - started if started is not None else None
        for node in result["nodes"].values():
            if "start" in node and "end" in node:
                node["seconds"] = node["end"] - node["start"]
        result["outputs"] = (await self.get(f"/history/{prompt_id}")).get(prompt_id, {}).get("outputs", {})
        return result


def summarize(variant):
    ok = [r for r in variant["runs"] if not r["error"]]
    warm = [r for r in ok if not r["cold"]]
    variant["warm_median"] = statistics.median(r["total"] for r in warm) if warm else None
    variant["cold"] = next((r["total"] for r in ok if r["cold"]), None)
    variant["vram_peak"] = {}
    for r in variant["runs"]:
        for index, used in r["vram_peak"].items():
            variant["vram_peak"][index] = max(variant["vram_peak"].get(index, 0), used)
    node_times = {}
    for r in warm:
        for node_id, node in r["nodes"].items():
            if "seconds" in node:
                node_times.setdefault((node_id, node["class_type"]), []).append(node["seconds"])
    variant["node_medians"] = {f"{class_type} #{node_id}": statistics.median(t) for (node_id, class_type), t in node_times.items()}
    variant["error"] = next((r["error"] for r in variant["runs"] if r["error"]), None)
    files = [output_files(r["outputs"]) for r in warm[:1]]
    variant["files"] = files[0] if files else []


def seconds(value):
    return "-" if value is None else f"{value:.1f} s"


def results_markdown(report):
    gpus = report["gpus"]
    primary = f"gpu:{gpus[0]['index']}"
    lines = [f"Workflow: `{report['workflow']}`" + (f" with {', '.join(report['set'])}" if report["set"] else "") + f". Server: {report['server']}, {report['date']}.",
             f"Each variant: 1 cold run after unloading models and clearing the cache, then {report['runs']} warm run(s) that only change the seed. Run k uses the same seed in every variant.", "",
             "## GPUs", ""]
    for i, d in enumerate(gpus):
        idle = report["idle_vram"].get(d["index"], 0)
        lines.append(f"- gpu:{d['index']} {gpu_name(d)}, {d['vram_total'] / GIB:.1f} GiB, {idle / GIB:.1f} GiB in use with no models loaded" + (" (primary)" if i == 0 else ""))
    if report.get("notes"):
        lines += ["", f"Notes: {report['notes']}"]
    lines += ["", "## Model loaders", ""]
    lines += [f"- `{l['label']}` ({l['class_type']} #{l['id']})" for l in report["loaders"]]

    variants = report["variants"]
    baseline = variants[0]["warm_median"]
    ranked = sorted(enumerate(variants, 1), key=lambda iv: (iv[1]["warm_median"] is None, iv[1]["warm_median"] or 0))
    peak_headers = [f"Peak gpu:{d['index']}" for d in gpus]
    lines += ["", "## Variants, fastest first", "",
              "| # | Variant | Warm median | Speedup | Cold | " + " | ".join(peak_headers) + " |",
              "|---|---|---|---|---|" + "---|" * len(gpus)]
    for i, v in ranked:
        speedup = f"{baseline / v['warm_median']:.2f}x" if baseline and v["warm_median"] else "-"
        name = v["name"] + (" *" if v["cross_card"] else "") + (f" (failed: {v['error']})" if v["error"] else "")
        peaks = " | ".join(f"{v['vram_peak'].get(d['index'], 0) / GIB:.1f} GiB" for d in gpus)
        lines.append(f"| {i} | {name} | {seconds(v['warm_median'])} | {speedup} | {seconds(v['cold'])} | {peaks} |")
    if any(v["cross_card"] for v in variants):
        lines += ["", f"\\* Samples on a GPU other than {primary}. The same seed gives a different image there."]
    if report["runs"] < 3:
        lines += ["", "With fewer than 3 warm runs the totals are noisy, for example when a prompt enhancer's output length varies. Compare the node table."]

    slowest = sorted((kv for kv in variants[0]["node_medians"].items() if kv[1] >= 0.1), key=lambda kv: -kv[1])[:10]
    if slowest:
        lines += ["", "## Slowest nodes, warm median in seconds (columns are variant numbers)", "",
                  "| Node | " + " | ".join(str(i) for i in range(1, len(variants) + 1)) + " |",
                  "|---|" + "---|" * len(variants)]
        for node, _ in slowest:
            cells = " | ".join(f"{v['node_medians'][node]:.1f}" if node in v["node_medians"] else "-" for v in variants)
            lines.append(f"| {node} | {cells} |")
    return "\n".join(lines)


def write_reports(report, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    results = results_markdown(report)
    outputs = ["", "## Outputs of warm run 1", ""] + [f"{i}. {v['name']}: {', '.join(v['files']) or '-'}" for i, v in enumerate(report["variants"], 1)]
    (out_dir / "report.md").write_text(f"# MultiGPU benchmark\n\n{results}\n" + "\n".join(outputs) + "\n", encoding="utf-8")
    request = f"{results}\n\n## Plan schema\n\n```json\n{PLAN_SCHEMA}\n```\n"
    (out_dir / "advisor_prompt.md").write_text(f"# System prompt\n\n{SYSTEM_PROMPT}\n\n# Prompt\n\n{request}", encoding="utf-8")
    return request


async def ask_advisor(server, args, request):
    graph = {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": args.advisor, "type": args.advisor_type, "device": "default"}},
        "2": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": SYSTEM_PROMPT}},
        "3": {"class_type": "TextGenerateV2", "inputs": {"clip": ["1", 0], "prompt": request, "max_length": args.advisor_tokens, "sampling_mode": "off",
                                                       "thinking": False, "use_default_template": True, "mtp": "auto", "system_prompt": ["2", 0]}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}},
    }
    await server.free()
    run = await server.run(graph)
    if run["error"]:
        raise SystemExit(f"Advisor failed: {run['error']}")
    return run["outputs"]["4"]["text"][0]


async def advise_report(args):
    report_dir = Path(args.workflow)
    request = (report_dir / "advisor_prompt.md").read_text(encoding="utf-8").split("\n\n# Prompt\n\n", 1)[1]
    async with aiohttp.ClientSession() as session:
        server = Server(session, args.server)
        await server.connect()
        reply = await ask_advisor(server, args, request)
    (report_dir / "advisor_reply.md").write_text(reply, encoding="utf-8")
    print(reply)


async def benchmark(args, graph, out_dir):
    async with aiohttp.ClientSession() as session:
        server = Server(session, args.server)
        gpus = await server.gpus()
        object_info = await server.get("/object_info")
        loaders = find_loaders(graph, object_info)
        variants = [("baseline", {})] + ([parse_variant(v, loaders) for v in args.variant] if args.variant else auto_variants(loaders, gpus))
        if args.list:
            print("GPUs:\n" + "\n".join(f"  gpu:{d['index']} {gpu_name(d)}" for d in gpus))
            print("Loaders:\n" + "\n".join(f"  {l['label']}  ({l['class_type']} #{l['id']})" for l in loaders))
            print("Variants:\n" + "\n".join(f"  {name}: {placement}" for name, placement in variants))
            return
        if len(gpus) < 2 and len(variants) > 1:
            raise SystemExit("The server sees one GPU. Start it with --default-device N instead of --cuda-device N.")
        missing = [c for c, _ in SELECT_NODES.values() if c not in object_info]
        if missing:
            raise SystemExit(f"The server has no {', '.join(missing)} node.")
        keys = seed_inputs(graph)
        if not keys:
            print("No seed inputs found, so warm runs will mostly hit the cache.")
        seed = graph[keys[0][0]]["inputs"][keys[0][1]] if keys else 0
        primary = f"gpu:{gpus[0]['index']}"

        await server.connect()
        results = []
        report = {"workflow": Path(args.workflow).name, "server": args.server, "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                  "set": args.set, "runs": args.runs, "notes": args.notes, "gpus": gpus, "loaders": loaders, "variants": results}
        for i, (name, placement) in enumerate(variants, 1):
            print(f"[{i}/{len(variants)}] {name}")
            idle_vram = await server.free(report.get("idle_vram"))
            if i == 1:
                report["idle_vram"] = idle_vram
            variant_graph = apply_placement(graph, loaders, placement)
            runs = []
            for k in range(args.runs + 1):
                timeout = args.timeout
                if timeout is None and results and results[0]["cold"] is not None:
                    timeout = 3 * results[0]["cold"] + 60
                run = await server.run(with_seed(variant_graph, keys, seed + k), timeout)
                run["cold"] = k == 0
                print(f"  {'cold' if k == 0 else 'warm'}: {seconds(run['total'])}" + (f"  failed: {run['error']}" if run["error"] else ""))
                runs.append(run)
                if run["error"]:
                    break
            cross_card = any(l["type"] == "MODEL" and placement.get(l["label"], "default") not in ("default", primary) for l in loaders)
            variant = {"name": name, "placement": placement, "cross_card": cross_card, "runs": runs}
            summarize(variant)
            results.append(variant)
            # written after every variant so a stopped sweep still leaves a report
            request = write_reports(report, out_dir)
        print(f"\n{results_markdown(report)}\n\nReports written to {out_dir}")
        if args.advisor:
            print("Asking the advisor model")
            reply = await ask_advisor(server, args, request)
            (out_dir / "advisor_reply.md").write_text(reply, encoding="utf-8")
            print(f"\n{reply}")


async def two_workers(args, graph, out_dir):
    async with aiohttp.ClientSession() as session:
        servers = [Server(session, address.strip()) for address in args.servers.split(",")]
        names = [", ".join(gpu_name(d) for d in await s.gpus()) for s in servers]
        for s in servers:
            await s.connect()
        keys = seed_inputs(graph)
        if not keys:
            raise SystemExit("Two-worker mode needs a seed input so every job runs.")
        seed = graph[keys[0][0]]["inputs"][keys[0][1]]

        print("Warm-up run on each server")
        await asyncio.gather(*(run_ok(s, with_seed(graph, keys, seed)) for s in servers))

        print(f"{args.jobs} jobs on {servers[0].address}")
        start = time.perf_counter()
        single_jobs = [(await run_ok(servers[0], with_seed(graph, keys, seed + 1 + k)))["total"] for k in range(args.jobs)]
        single = time.perf_counter() - start

        print(f"{args.jobs} jobs across {len(servers)} servers, each taking the next job when it is free")
        queue = asyncio.Queue()
        for k in range(args.jobs):
            queue.put_nowait(seed + 1 + args.jobs + k)
        job_times = {s.address: [] for s in servers}

        async def worker(server):
            while not queue.empty():
                job_times[server.address].append((await run_ok(server, with_seed(graph, keys, queue.get_nowait())))["total"])

        start = time.perf_counter()
        await asyncio.gather(*(worker(s) for s in servers))
        parallel = time.perf_counter() - start

    # Wall time over a few jobs is set by whichever server finishes last; the per-job medians give the long-queue rate.
    steady = sum(60 / statistics.median(t) for t in job_times.values() if t)
    lines = [f"# Two-worker benchmark\n\nWorkflow: `{Path(args.workflow).name}`, {args.jobs} jobs.\n",
             "| Setup | Jobs | Median job | Wall time | Jobs per minute |", "|---|---|---|---|---|",
             f"| {servers[0].address} ({names[0]}) alone | {args.jobs} | {statistics.median(single_jobs):.1f} s | {single:.1f} s | {60 * args.jobs / single:.2f} |"]
    lines += [f"| {s.address} ({n}) with the others running | {len(job_times[s.address])} | {seconds(statistics.median(job_times[s.address]) if job_times[s.address] else None)} | | |"
              for s, n in zip(servers, names)]
    lines += [f"| All servers | {args.jobs} | | {parallel:.1f} s | {60 * args.jobs / parallel:.2f} |",
              f"\nSpeedup over this run: {single / parallel:.2f}x. Long queue: {steady:.2f} jobs per minute, {steady * statistics.median(single_jobs) / 60:.2f}x. The same seed gives a different image on each GPU."]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "two_workers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines) + f"\n\nReport written to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark a workflow with its models placed on different GPUs.")
    parser.add_argument("workflow", help="API-format .json, a .png or video saved by ComfyUI, or a report directory to ask the --advisor about again.")
    parser.add_argument("--server", default="127.0.0.1:8188")
    parser.add_argument("--runs", type=int, default=3, help="Warm runs per variant, after one cold run.")
    parser.add_argument("--variant", action="append", default=[], metavar="SPEC",
                        help="Placement to test instead of the automatic sweep, for example 'clip:*PE*=gpu:1,vae:*=gpu:1'. Repeatable.")
    parser.add_argument("--set", action="append", default=[], metavar="NODE.INPUT=VALUE",
                        help="Change an input before benchmarking, for example 160.value=8 to cut the steps of a long workflow. Repeatable.")
    parser.add_argument("--timeout", type=float, default=None, help="Stop a run after this many seconds. Default: 3x the baseline's cold run plus a minute.")
    parser.add_argument("--list", action="store_true", help="Show GPUs, loaders and variants, then exit.")
    parser.add_argument("--notes", default="", help="Extra hardware facts for the advisor, such as PCIe links.")
    parser.add_argument("--out", type=Path, default=None, help="Report directory. Default: output/multigpu_bench/<workflow>-<time>.")
    parser.add_argument("--advisor", metavar="TEXT_ENCODER", help="Text encoder file to ask for a plan with Generate Text.")
    parser.add_argument("--advisor-type", default="qwen_image", help="CLIPLoader type of the advisor model.")
    parser.add_argument("--advisor-tokens", type=int, default=2048)
    parser.add_argument("--servers", help="Comma-separated single-GPU servers to compare one worker against all of them.")
    parser.add_argument("--jobs", type=int, default=4, help="Jobs per setup in --servers mode.")
    args = parser.parse_args()

    if Path(args.workflow).is_dir():
        if not args.advisor:
            raise SystemExit("Pass --advisor to ask for a plan from an existing report.")
        asyncio.run(advise_report(args))
        return
    graph = load_workflow(args.workflow)
    set_inputs(graph, args.set)
    out_dir = args.out or Path(__file__).resolve().parent.parent / "output" / "multigpu_bench" / f"{Path(args.workflow).stem}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    asyncio.run(two_workers(args, graph, out_dir) if args.servers else benchmark(args, graph, out_dir))


if __name__ == "__main__":
    main()
