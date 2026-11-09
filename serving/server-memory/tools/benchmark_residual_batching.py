#!/usr/bin/env python3
"""Benchmark the real encoder on synthetic captured features, on a queue GPU.

Includes CPU feature staging and encoder forward; excludes the actor prefill
and summary generation. Does not launch/restart an inference server or train.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/recursive_memory/stack/payload/workspace"))
import torch
from context_compression.qwen38_residual_serving_contract import load_residual_serving_contract
from context_compression.remory_public_release import serving_weight_path
from context_compression.qwen38_residual_sglang_hook import (
    _load_summary_residual_compressor, _residual_microbatch_limit, encode_residual_blocks,
)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", choices=["1", "2", "4", "8", "auto"],
                        default=["1", "2", "4", "8", "auto"])
    args = parser.parse_args()
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not devices or len(devices.split(",")) != 1:
        raise RuntimeError("select exactly one GPU allocated for this benchmark with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    torch.manual_seed(20261109)
    raw = load_residual_serving_contract(args.checkpoint, verify_weights=True)
    metadata = SimpleNamespace(raw_config=raw, checkpoint=args.checkpoint,
                               state_path=serving_weight_path(args.checkpoint))
    started = time.monotonic()
    model = _load_summary_residual_compressor(metadata, torch.device("cuda:0"), torch.bfloat16)
    report = dict(checkpoint=str(args.checkpoint), devices=devices,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        load_seconds=time.monotonic() - started,
        scope="synthetic captured CPU features + real frozen encoder; excludes actor prefill/summary generation",
        cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    hidden = raw["compressor"]["target_hidden_size"]
    layers = len(raw["compressor"]["target_layer_ids"])
    for count, summary_rows, mixed in [(16, 256, False), (4, 4097, False), (8, 256, True)]:
        start, end = 7, 7 + (count - 1) * 1024 + 17
        aux = [torch.randn(end, hidden, dtype=torch.bfloat16) for _ in range(layers)]
        summary = torch.randn(1, summary_rows, hidden, dtype=torch.bfloat16,
                              device="cpu" if summary_rows > 4096 else "cuda")
        summary_mask = torch.ones(summary.shape[:2], dtype=torch.bool, device=summary.device)
        depths = [1] * (count // 2) + [0] * (count // 2) if mixed else [0] * count

        @torch.inference_mode()
        def serial():
            pieces = []
            for i, left in enumerate(range(start, end, 1024)):
                right = min(end, left + 1024)
                features = torch.cat([h[left:right] for h in aux], -1).cuda()[None]
                encoded = model(features, torch.ones(features.shape[:2], dtype=torch.bool, device="cuda"),
                    source_token_lengths=torch.tensor([right - left], device="cuda"),
                    slot_token_lengths=torch.tensor([1024], device="cuda"),
                    summary_features=summary, summary_attention_mask=summary_mask, input_depth=depths[i])
                pieces.append(encoded.embeddings[encoded.attention_mask])
            return torch.cat(pieces)

        reference = serial().float().cpu()
        automatic = _residual_microbatch_limit(model, aux, summary, 1024, 16)
        case = dict(blocks=count, source_tokens=end-start, summary_rows=summary_rows,
                    mixed_depth=mixed, automatic_limit=automatic, measurements=[])
        report["cases"].append(case)
        for requested in args.batch_sizes:
            size = None if requested == "auto" else int(requested)
            fn = serial if size == 1 else lambda: encode_residual_blocks(model, aux,
                start=start, end=end, block=1024, ratio=16, summary=summary, summary_mask=summary_mask,
                input_depths=depths, max_batch_blocks=size)
            result = fn()  # warm up kernels for this shape
            actual = result.float().cpu()
            del result
            delta = actual - reference
            nrms = float(delta.square().mean().sqrt() / reference.square().mean().sqrt())
            cosine = float(torch.nn.functional.cosine_similarity(actual, reference, dim=-1).min())
            max_abs = float(delta.abs().max())
            # BF16 changes GEMM reduction order with batch geometry. Report the
            # errors, never describe this as bitwise identity or quality eval.
            if not torch.isfinite(actual).all() or nrms > 0.02 or cosine < 0.999:
                raise AssertionError(f"encoder numerical drift: NRMS={nrms}, min cosine={cosine}")
            walls, gpu_times, peaks = [], [], []
            for _ in range(3):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                tick = time.monotonic()
                begin.record()
                result = fn()
                finish.record()
                torch.cuda.synchronize()
                walls.append(time.monotonic() - tick)
                gpu_times.append(begin.elapsed_time(finish))
                peaks.append((torch.cuda.max_memory_allocated() - baseline) / (1 << 20))
                del result
            measurement = dict(batch_blocks="auto" if size is None else size,
                wall_seconds=walls, median_seconds=statistics.median(walls),
                cuda_elapsed_ms=gpu_times, transient_peak_mib=max(peaks),
                normalized_rmse=nrms, min_slot_cosine=cosine, max_abs_error=max_abs)
            case["measurements"].append(measurement)
            save()
            print(json.dumps(dict(blocks=count, summary_rows=summary_rows, **measurement)), flush=True)
        del aux, summary, reference, actual
    report["complete"] = True
    save()


if __name__ == "__main__":
    run()
