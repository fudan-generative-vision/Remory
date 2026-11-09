"""Reference-only interactive checkpoints, including one-time legacy import."""
from context_compression.remory_memory import RemoteMemory

TAG = "__remory_remote_snapshot_v1__"


def map_values(value, transform):
    replacement = transform(value)
    if replacement is not value:
        return replacement
    if isinstance(value, dict):
        return {k: map_values(v, transform) for k, v in value.items()}
    if isinstance(value, list):
        return [map_values(v, transform) for v in value]
    if isinstance(value, tuple):
        return tuple(map_values(v, transform) for v in value)
    return value


def pack_memory_state(state, client):
    snapshots = set()
    def pack(value):
        if not isinstance(value, RemoteMemory):
            return value
        saved = client.save_memory(value)
        snapshots.add(saved["snapshot"])
        return {TAG: saved, "backend": value.backend}
    return map_values(state, pack), snapshots


def restore_memory_state(state, client):
    import torch
    loaded = {}
    snapshots = set()
    before = set(client.owned_memory())
    def restore(value):
        if isinstance(value, dict) and TAG in value:
            if set(value) != {TAG, "backend"} or value["backend"] != client.base_url:
                raise ValueError("checkpoint must resume on its original memory backend")
            saved = value[TAG]
            key = saved["snapshot"]
            snapshots.add(key)
            if key not in loaded:
                loaded[key] = client.load_memory(saved)
            return loaded[key]
        if isinstance(value, torch.Tensor):
            if value.ndim != 2 or not value.is_floating_point():
                raise ValueError("legacy checkpoint contains an unsupported tensor")
            key = ("legacy", id(value))
            if key not in loaded:
                loaded[key] = client.import_memory(value)
            return loaded[key]
        return value
    try:
        result = map_values(state, restore)
        client._checkpoint_snapshots = snapshots
        return result
    except BaseException:
        owned = client.owned_memory()
        for key in set(owned) - before:
            try:
                client.release_memory(owned[key])
            except Exception:
                pass  # retained in the client's retry queue
        raise


def load_local_snapshot(remote):
    """Training-side reader, run on the inference host with its snapshot mount."""
    import os
    import re
    from pathlib import Path
    import torch
    saved = remote["snapshot"]
    key, hook = saved["snapshot"], saved["hook_abi_digest"]
    if not re.fullmatch(r"[0-9a-f]{64}", key) or not re.fullmatch(r"sha256:[0-9a-f]{64}", hook):
        raise ValueError("invalid remote snapshot identity")
    root = os.environ.get("REMORY_MEMORY_DIR")
    if not root:
        raise RuntimeError("remote-memory training must run beside the inference engine with REMORY_MEMORY_DIR")
    path = Path(root) / hook[7:] / "tp-0" / (key + ".pt")
    data = torch.load(path, map_location="cpu", weights_only=True)
    tensor = data["tensor"]
    if (data["hook_abi_digest"] != hook or list(tensor.shape) != saved["shape"]
            or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all())):
        raise ValueError("invalid saved remote memory")
    return tensor


def save_reference_checkpoint(path, state, client):
    """Publish references durably before retiring the previous snapshots.

    Keep a copy of the old checkpoint when migrating a legacy tensor state.
    A caller with a ResidualGenerator invokes commit_remote_retirements only
    after this function returns. Never unpickle checkpoints from other users.
    """
    import os
    import pickle
    import tempfile
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise ValueError("checkpoint cannot be a symlink")
    packed, snapshots = pack_memory_state(state, client)
    payload = dict(schema="remory_reference_checkpoint_v1", backend=client.base_url, state=packed)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".remory-checkpoint-", delete=False) as f:
            name = f.name
            os.fchmod(f.fileno(), 0o600)
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        name = None
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if name is not None:
            os.unlink(name)
    client.checkpoint_saved(snapshots)


def load_reference_checkpoint(path, client):
    """Restore only a private, owned checkpoint on its saved backend."""
    import os
    import pickle
    from pathlib import Path
    path = Path(path)
    if path.is_symlink() or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError("checkpoint must be a private owned file")
    payload = pickle.loads(path.read_bytes())
    if payload.get("schema") != "remory_reference_checkpoint_v1" or payload.get("backend") != client.base_url:
        raise ValueError("checkpoint must resume on its original pinned backend")
    return restore_memory_state(payload["state"], client)
