"""Exercise the paths that a greedy server warmup does not cover."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .deploy import atomic, environment

KERNEL_CHECK = r"""
import json, sys, torch
from sgl_kernel import silu_and_mul
from flashinfer.sampling import top_k_top_p_sampling_from_probs
config = json.load(open(sys.argv[1]))
vocab = config.get("text_config", config)["vocab_size"]
torch.manual_seed(2320)
x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
output = torch.empty(4, 128, device="cuda", dtype=torch.bfloat16)
silu_and_mul(x, output)
torch.testing.assert_close(output, torch.nn.functional.silu(x[:, :128]) * x[:, 128:], rtol=0.03, atol=0.03)
probs = torch.randn(8, vocab, device="cuda", dtype=torch.float32).softmax(-1)
top_k = torch.full((8,), 20, device="cuda", dtype=torch.int32)
top_p = torch.full((8,), 0.95, device="cuda", dtype=torch.float32)
sample = top_k_top_p_sampling_from_probs(probs, top_k, top_p, filter_apply_order="joint", check_nan=True)
torch.cuda.synchronize()
assert sample.shape == (8,)
assert bool((probs.topk(20, dim=-1).indices == sample[:, None]).any(dim=-1).all())
print(json.dumps(dict(status="ok", capability=torch.cuda.get_device_capability(),
                     torch=torch.__version__, vocab_size=vocab, batch_size=8,
                     silu_and_mul="ok", top_k=20, top_p=0.95, sampling="ok")))
"""


def kernels(state):
    from .gpu import selected_devices

    env = environment(state)
    mask = selected_devices(state["server"], env)
    if mask is not None:
        env["CUDA_VISIBLE_DEVICES"] = mask
    result = subprocess.run(
        [
            str(Path(state["prefix"]) / "server/bin/python"),
            "-c",
            KERNEL_CHECK,
            str(Path(state["target"]) / "config.json"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode:
        raise RuntimeError(
            "GPU sampling/kernel acceptance failed:\n" + result.stderr[-8000:]
        )
    report = json.loads(result.stdout.splitlines()[-1])
    atomic(Path(state["prefix"]) / "receipts/kernels.json", report)
    return report


def structured(reply):
    if reply.get("isError"):
        raise RuntimeError("document tool acceptance returned an error")
    if isinstance(reply.get("structuredContent"), dict):
        return reply["structuredContent"]
    for block in reply.get("content", []):
        if block.get("type") == "text":
            try:
                result = json.loads(block["text"])
            except (ValueError, KeyError):
                continue
            if isinstance(result, dict):
                return result
    raise RuntimeError("document tool returned no structured result")


def documents(state):
    from shared_runtime.runner import preflight
    from terminal_bench_eval.document_tools import DocumentTools

    # MCP discovery and local Markdown conversion do not use cloud APIs. Keep
    # this acceptance independent of paid credentials and HF model downloads.
    names = ("SERPER_API_KEY", "MINERU_API_KEY", "HF_HUB_OFFLINE")
    previous = {key: os.environ.get(key) for key in names}
    try:
        os.environ.update(
            SERPER_API_KEY="local-acceptance-no-api-call",
            MINERU_API_KEY="local-acceptance-no-api-call",
            HF_HUB_OFFLINE="1",
        )
        preflight()
        with tempfile.TemporaryDirectory(prefix="rm-documents-acceptance-") as name:
            root = Path(name)
            rootfs, artifacts = root / "rootfs", root / "artifacts"
            (rootfs / "tmp").mkdir(parents=True)
            (rootfs / "tmp/input.md").write_text(
                "# Acceptance\n\nThe test value is 2320.\n"
            )
            tools = DocumentTools(rootfs, artifacts)
            try:
                tools.start()
                docling = next(s for s in tools.sessions if s.name == "docling")
                converted = structured(
                    docling.request(
                        "tools/call",
                        name="convert_document_into_docling_document",
                        arguments={"source": "/tmp/input.md"},
                    )
                )
                exported = structured(
                    docling.request(
                        "tools/call",
                        name="export_docling_document_to_markdown",
                        arguments={"document_key": converted["document_key"]},
                    )
                )
                if "2320" not in exported["markdown"]:
                    raise RuntimeError("document roundtrip lost its synthetic canary")
                report = {
                    "status": "ok",
                    "mcp_servers": [s.name for s in tools.sessions],
                    "docling_markdown_roundtrip": "ok",
                    "cloud_api_calls": False,
                }
            finally:
                tools.close()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    atomic(Path(state["prefix"]) / "receipts/documents.json", report)
    return report
