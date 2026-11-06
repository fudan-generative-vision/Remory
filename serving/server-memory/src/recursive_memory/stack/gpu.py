"""Resolve GPUs at launch time, using the serving Python's CUDA view."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys

SUPPORTED_TP = {1, 2, 4, 8}
PROBE = """
import json, torch
rows = []
for index in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(index)
    rows.append(dict(name=p.name, uuid='GPU-' + str(p.uuid).removeprefix('GPU-'), memory_bytes=p.total_memory,
                     capability=[p.major, p.minor]))
print(json.dumps(rows))
"""


def validate_devices(value):
    if value == "auto":
        return
    if not isinstance(value, str) or not value:
        raise ValueError("CUDA_VISIBLE_DEVICES is empty: no GPU was selected")
    parts = value.split(",")
    if any(not re.fullmatch(r"(?:[0-9]+|GPU-[a-fA-F0-9-]+)", p) for p in parts):
        raise ValueError(
            "use comma-separated GPU ordinals or GPU UUIDs; MIG is not supported"
        )
    normalized = [str(int(p)) if p.isdigit() else p.lower() for p in parts]
    if len(set(normalized)) != len(parts):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate GPUs")


def selected_devices(server, env):
    if "CUDA_VISIBLE_DEVICES" in env:
        value = env["CUDA_VISIBLE_DEVICES"]
        alias = env.get("CUDA_VISIBLE_DEVICE")
        if alias is not None and alias != value:
            raise ValueError("CUDA_VISIBLE_DEVICE conflicts with CUDA_VISIBLE_DEVICES")
    elif "CUDA_VISIBLE_DEVICE" in env:
        value = env["CUDA_VISIBLE_DEVICE"]
        print(
            "[GPU] CUDA_VISIBLE_DEVICE is an alias; prefer CUDA_VISIBLE_DEVICES",
            file=sys.stderr,
        )
    else:
        value = server.get("devices", "auto")
    validate_devices(value)
    return None if value == "auto" else value


def discover(python, devices, env):
    probe_env = dict(env)
    probe_env.pop("CUDA_VISIBLE_DEVICE", None)
    if devices is None:
        probe_env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        probe_env["CUDA_VISIBLE_DEVICES"] = devices
    result = subprocess.run(
        [str(python), "-c", PROBE],
        env=probe_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or not rows:
        raise ValueError("the serving runtime sees no CUDA GPUs")
    if devices is not None and len(rows) != len(devices.split(",")):
        raise ValueError(
            "CUDA did not resolve every requested GPU; refusing a truncated selection"
        )
    return rows


def resolve(server, python, *, env=None, probe=None):
    env = os.environ if env is None else env
    probe = probe or discover
    devices = selected_devices(server, env)
    rows = probe(python, devices, env)
    count = len(rows)
    if count not in SUPPORTED_TP:
        raise ValueError(
            f"{count} visible GPUs: this Qwen stack supports TP1/2/4/8; select a supported count with CUDA_VISIBLE_DEVICES"
        )
    uuids = [row["uuid"] for row in rows]
    if len(set(uuids)) != count or any(not value for value in uuids):
        raise ValueError("CUDA resolved duplicate/invalid GPU identities")
    if any(row["capability"][0] < 8 for row in rows):
        raise ValueError("BF16/FlashInfer requires Ampere-or-newer CUDA devices")
    # An explicit runtime mask overrides an old installation's topology pin.
    runtime_mask = "CUDA_VISIBLE_DEVICES" in env or "CUDA_VISIBLE_DEVICE" in env
    tp = count if runtime_mask else server.get("tp", "auto")
    if tp != "auto" and tp != count:
        raise ValueError(
            "configured TP must match all selected GPUs; do not silently drop devices"
        )
    smallest = min(row["memory_bytes"] for row in rows) / 2**30
    # BF16 27B target is sharded; Compressor VRAM is replicated when enabled.
    compressor_gib = server.get("compressor_memory_gib", 8)
    if type(compressor_gib) not in (int, float) or not math.isfinite(compressor_gib) or compressor_gib < 0:
        raise ValueError("compressor_memory_gib must be finite and nonnegative")
    required = 54 / count + compressor_gib + 3
    fraction = server.get("memory_fraction", "auto")
    if fraction == "auto":
        fraction = max(0.55, math.ceil(required / smallest * 100) / 100)
        if fraction > 0.85:
            raise ValueError(
                "insufficient conservative VRAM headroom for BF16 target and configured Compressor; use more/larger GPUs"
            )
    if not 0.1 <= fraction <= 0.9 or smallest * fraction < 54 / count + compressor_gib:
        raise ValueError(
            "memory_fraction leaves insufficient space for estimated model weights"
        )
    large_capacity = smallest >= 128 or (smallest >= 64 and count >= 2)
    default_context = 262144 if large_capacity else 65536
    effective = {**server, "devices": devices, "tp": count, "memory_fraction": fraction}
    for name in ("context_length", "max_total_tokens"):
        effective[name] = server.get(name, "auto")
        if effective[name] == "auto":
            effective[name] = default_context
    effective["cuda_graph"] = server.get("cuda_graph", "auto")
    if effective["cuda_graph"] == "auto":
        effective["cuda_graph"] = large_capacity
    pcie_geforce = any("GeForce" in row["name"] for row in rows)
    effective["nccl_p2p_disable"] = env.get(
        "NCCL_P2P_DISABLE", "1" if pcie_geforce else None
    )
    effective["gpu_inventory"] = rows
    return effective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python", required=True, help="Python from the serving environment"
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="print validated exports for an existing launcher",
    )
    args = parser.parse_args()
    server = {
        "devices": "auto",
        "tp": os.environ.get("TERMINAL_BENCH_POOL_TP_SIZE", "auto"),
    }
    if server["tp"] != "auto":
        server["tp"] = int(server["tp"])
    for key, variable, convert in (
        ("memory_fraction", "MEM_FRACTION_STATIC", float),
        ("context_length", "CONTEXT_LENGTH", int),
        ("max_total_tokens", "MAX_TOTAL_TOKENS", int),
    ):
        if variable in os.environ:
            server[key] = convert(os.environ[variable])
    result = resolve(server, args.python)
    if not args.shell:
        print(json.dumps(result, indent=2))
        return
    exports = {
        "TERMINAL_BENCH_POOL_TP_SIZE": result["tp"],
        "MEM_FRACTION_STATIC": result["memory_fraction"],
        "CONTEXT_LENGTH": result["context_length"],
        "MAX_TOTAL_TOKENS": result["max_total_tokens"],
        "ENABLE_CUDA_GRAPH": os.environ.get(
            "ENABLE_CUDA_GRAPH", str(int(result["cuda_graph"]))
        ),
    }
    # Keep the exact mask/order, or pin the discovered UUIDs when using all.
    exports["CUDA_VISIBLE_DEVICES"] = result["devices"] or ",".join(
        row["uuid"] for row in result["gpu_inventory"]
    )
    if result["nccl_p2p_disable"] is not None:
        exports["NCCL_P2P_DISABLE"] = result["nccl_p2p_disable"]
    for key, value in exports.items():
        print(f"export {key}={shlex.quote(str(value))}")
    print(
        f"[GPU] {result['tp']} visible devices; TP={result['tp']} context={result['context_length']} token_pool={result['max_total_tokens']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
