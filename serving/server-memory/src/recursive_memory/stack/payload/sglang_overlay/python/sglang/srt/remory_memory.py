"""TP-coordinated GPU memory operations. No tensor is returned over HTTP."""
import uuid
import os
import re
from pathlib import Path
import torch

from context_compression.remory_memory import digest
from context_compression.sglang_compressor_hook import compute_memory_handle_identity


def collective_publish(registry, *, tensor, op_id, request_digest, gather):
    descriptor = None
    try:
        descriptor = registry.publish(op_id=op_id,
            memory_identity=compute_memory_handle_identity(server_epoch=registry.server_epoch,
                request_digest=request_digest, hook_abi_digest=registry.hook_abi_digest),
            request_digest=request_digest, tensor=tensor.detach().contiguous())
        outcome = dict(success=True, handle=descriptor)
    except Exception as exc:
        outcome = dict(success=False, message=str(exc))
    results = gather(outcome)
    if not results or not all(r == results[0] and r.get("success") for r in results):
        if descriptor is not None:
            registry.release(op_id="rollback." + uuid.uuid4().hex, handle=descriptor)
        raise RuntimeError("TP memory publication failed: " + repr(results))
    return descriptor


def memory_operation(registry, payload, *, gather):
    """Compose slices on device or renew a reference, with uniform TP errors.

    Scheduler requests hold strong tensor references, so releasing a registry
    entry never frees storage still used by admitted model forwards.
    """
    operation = payload.get("operation")
    if operation in {"save", "load", "delete_snapshot", "renew", "import"}:
        return lifecycle_operation(registry, payload, gather=gather)
    pieces = []
    try:
        if payload.get("operation") != "compose" or set(payload) != {"operation", "op_id", "parts"}:
            raise ValueError("unsupported Remory memory operation")
        if not isinstance(payload["parts"], list) or not payload["parts"]:
            raise ValueError("composition requires memory parts")
        for part in payload["parts"]:
            if set(part) != {"handle", "start", "stop"}:
                raise ValueError("invalid memory slice")
            tensor = registry.resolve(part["handle"]).tensor
            a, b = part["start"], part["stop"]
            if type(a) is not int or type(b) is not int or not 0 <= a < b <= tensor.shape[0]:
                raise ValueError("invalid memory slice coordinates")
            pieces.append(tensor[a:b])
        # Clone even one slice: published storage must not retain a large parent.
        tensor = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0].clone()
        outcome = dict(success=True, shape=list(tensor.shape), dtype=str(tensor.dtype))
    except Exception as exc:
        outcome = dict(success=False, message=str(exc))
    results = gather(outcome)
    if not all(r == results[0] and r.get("success") for r in results):
        raise RuntimeError("TP memory composition failed: " + repr(results))
    return {"handle": collective_publish(registry, tensor=tensor, op_id=payload["op_id"],
                              request_digest=digest(payload), gather=gather)}


def lifecycle_operation(registry, payload, *, gather):
    """Snapshots live beside the model engine, never on the calling machine."""
    operation = payload["operation"]
    try:
        if operation == "import":
            if set(payload) != {"operation", "op_id", "values"}:
                raise ValueError("invalid one-time legacy memory import")
            tensor = torch.tensor(payload["values"], dtype=torch.float32)
            if tensor.ndim != 2 or tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
                raise ValueError("invalid imported memory")
            if tensor.numel() * tensor.element_size() > registry.max_bytes:
                raise ValueError("legacy memory import exceeds registry capacity")
            tensor = tensor.to(device="cuda" if registry.require_cuda else "cpu")
            result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        elif operation == "renew":
            if set(payload) != {"operation", "handles"}:
                raise ValueError("invalid lease request")
            leases = [registry.renew(handle) for handle in payload["handles"]]
            result = {"leases": leases}
        else:
            root = os.environ.get("REMORY_MEMORY_DIR")
            if not root:
                raise ValueError("REMORY_MEMORY_DIR is required for remote recovery")
            directory = Path(root) / registry.hook_abi_digest[7:] / ("tp-" + str(registry.tp_rank))
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if operation == "save":
                if set(payload) != {"operation", "handle"}:
                    raise ValueError("invalid memory snapshot request")
                handle = payload["handle"]
                tensor = registry.resolve(handle).tensor
                key = digest(handle)[7:]
                path = directory / (key + ".pt")
                tmp = directory / (key + "." + uuid.uuid4().hex + ".tmp")
                try:
                    with tmp.open("xb") as stream:
                        os.chmod(tmp, 0o600)
                        torch.save(dict(tensor=tensor.detach().cpu(), hook_abi_digest=registry.hook_abi_digest,
                                        memory_identity=handle["memory_identity"]), stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(tmp, path)
                finally:
                    tmp.unlink(missing_ok=True)
                result = dict(snapshot=key, shape=list(tensor.shape), hook_abi_digest=registry.hook_abi_digest)
            elif operation in {"load", "delete_snapshot"}:
                required = {"operation", "snapshot"} | ({"op_id"} if operation == "load" else set())
                if set(payload) != required or not re.fullmatch(r"[0-9a-f]{64}", payload["snapshot"]):
                    raise ValueError("invalid snapshot identifier")
                path = directory / (payload["snapshot"] + ".pt")
                if operation == "delete_snapshot":
                    path.unlink(missing_ok=True)
                    result = {"deleted": True}
                else:
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if saved["hook_abi_digest"] != registry.hook_abi_digest:
                        raise ValueError("snapshot checkpoint identity mismatch")
                    tensor = saved["tensor"]
                    if tensor.ndim != 2 or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                        raise ValueError("invalid snapshot tensor")
                    tensor = tensor.to(device="cuda" if registry.require_cuda else "cpu")
                    result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        outcome = dict(success=True, result=result)
    except Exception as exc:
        outcome = dict(success=False, message=str(exc))
    results = gather(outcome)
    if not results or not all(r == results[0] and r.get("success") for r in results):
        raise RuntimeError("TP memory lifecycle failed: " + repr(results))
    if operation in {"load", "import"}:
        return {"handle": collective_publish(registry, tensor=tensor, op_id=payload["op_id"],
                                             request_digest=digest(payload), gather=gather)}
    return result
