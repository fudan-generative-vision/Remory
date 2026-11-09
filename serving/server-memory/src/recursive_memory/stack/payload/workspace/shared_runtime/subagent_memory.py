"""Immutable child checkpoints inserted at a parent's causal round boundary."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

POLICY = "subagent_completion_summary_residual_fold_v1"


def identity(generator):
    return hashlib.sha256(json.dumps(generator.deployment, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def export_checkpoint(backend, session, directory, *, child_id, parent_id, status="completed"):
    import torch
    from .storage import atomic_json
    generator = backend._generator
    messages = deepcopy(session.messages)
    if len(messages) % 2:
        messages.append({"role": "user", "content":
            "[Sub-agent execution ended. Preserve the execution evidence and final result for the parent.]"})
    if len(messages) <= 2:
        raise RuntimeError("sub-agent has no execution history to fold")
    summary = generator.summarize(messages=messages, episode_id=session.episode_id,
        round_index=len(messages) // 2, raw_suffix_images=tuple(session.raw_suffix_images))
    prepared = generator._prepared
    if prepared is None or prepared["memory"] is None:
        raise RuntimeError("sub-agent completion requires summary AND residual memory")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if getattr(generator, "server_memory", False):
        memory_record = dict(remote_memory=dict(backend=generator.client.base_url,
            snapshot=generator.client.save_memory(prepared["memory"], preserve=True)))
    else:
        temporary = directory / "memory.pt.tmp"
        torch.save({"memory": prepared["memory"].detach().cpu().contiguous()}, temporary)
        temporary.chmod(0o600)
        temporary.replace(directory / "memory.pt")
        memory_record = dict(memory_sha256=hashlib.sha256((directory / "memory.pt").read_bytes()).hexdigest())
    receipt = dict(policy=POLICY, child_id=child_id, parent_id=parent_id,
        status=status, model_identity=identity(generator), episode_id=session.episode_id,
        summary=summary, summary_ids=list(prepared["body"]), depths=list(prepared["depths"]),
        **memory_record)
    receipt["fold_id"] = hashlib.sha256(json.dumps(receipt, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    atomic_json(directory / "checkpoint.json", receipt)
    return str(directory / "checkpoint.json")


def read_checkpoint(path, generator, *, parent_id, child_id):
    import torch
    path = Path(path)
    receipt = json.loads(path.read_text())
    digest = receipt.pop("fold_id")
    if digest != hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        raise RuntimeError("sub-agent checkpoint receipt changed")
    if (receipt["policy"] != POLICY or receipt["parent_id"] != parent_id
            or receipt["child_id"] != child_id or receipt["model_identity"] != identity(generator)):
        raise RuntimeError("sub-agent checkpoint owner/model identity mismatch")
    if "remote_memory" in receipt:
        remote = receipt["remote_memory"]
        if not getattr(generator, "server_memory", False) or remote["backend"] != generator.client.base_url:
            raise RuntimeError("child memory requires the same pinned backend as its parent")
        memory = generator.client.load_memory(remote["snapshot"], preserve=True)
    else:
        memory_path = path.with_name("memory.pt")
        if hashlib.sha256(memory_path.read_bytes()).hexdigest() != receipt["memory_sha256"]:
            raise RuntimeError("sub-agent residual checksum mismatch")
        memory = torch.load(memory_path, weights_only=True, map_location="cpu")["memory"]
    body, depths = receipt["summary_ids"], receipt["depths"]
    if (not isinstance(body, list) or not body or any(type(v) is not int or v < 0 for v in body)
            or memory.ndim != 2 or memory.shape[1] != generator.deployment["compressor"]["target_hidden_size"]
            or memory.shape[0] <= 0
            or (generator.residual["memory_budget"] is not None and memory.shape[0] > generator.residual["memory_budget"])
            or (isinstance(memory, torch.Tensor)
                and (not memory.is_floating_point() or not bool(torch.isfinite(memory).all())))
            or len(depths) != memory.shape[0] or any(type(v) is not int or v <= 0 for v in depths)):
        if getattr(generator, "server_memory", False):
            generator.client.release_memory(memory)
        raise RuntimeError("invalid sub-agent summary/residual geometry")
    if getattr(generator, "server_memory", False) and isinstance(memory, torch.Tensor):
        memory = generator.client.import_memory(memory)
    return dict(receipt, fold_id=digest, memory=memory)


class SubagentMemoryMixin:
    def initialize_subagent_memory(self):
        self._folded_branches = []
        self._folded_ids = set()
        self._context_memory = None
        self._context_depths = []
        self._context_memory_key = None

    def add_subagent_checkpoint(self, checkpoint, *, messages, episode_id):
        if self.text_only:
            raise RuntimeError("sub-agent folding requires a residual backend")
        fold_id = checkpoint["fold_id"]
        if fold_id in self._folded_ids:
            if getattr(self, "server_memory", False):
                current = [self.memory, *(e["memory"] for e in self._folded_branches)]
                if all(checkpoint["memory"] is not v for v in current):
                    self.client.release_memory(checkpoint["memory"])
            return False
        _, prefix, closed = self._text_history(messages, episode_id)
        entry = dict(checkpoint, offset=len(closed))
        self._folded_branches.append(entry)
        try:
            ids = self._active_text_context(prefix, closed)
            ceiling = min(self.server_info[k] for k in
                ("context_length", "max_total_num_tokens", "max_req_input_len"))
            if len(ids) + self.config.capacity_pressure_reserve_tokens >= ceiling:
                raise RuntimeError("sub-agent fold requires parent compaction before admission")
        except BaseException:
            self._folded_branches.pop()
            try:
                self._active_text_context(prefix, closed)
            finally:
                if getattr(self, "server_memory", False):
                    # This candidate never entered the parent's checkpoint.
                    # Release its handle now; the child's preserved snapshot
                    # remains available for another admission attempt.
                    self.client.release_memory(checkpoint["memory"])
            raise
        self._folded_ids.add(fold_id)
        self._runtime_event(dict(event="subagent_memory_folded", policy=POLICY,
            episode_id=episode_id, child_id=entry["child_id"], fold_id=fold_id,
            parent_completed_rounds=(len(messages) - 2) // 2,
            summary_tokens=len(entry["summary_ids"]), memory_tokens=int(entry["memory"].shape[0]),
            parent_context_tokens=len(ids), trigger="child_completion", context_threshold=None))
        return True

    def _with_subagent_memory(self, prefix, closed, active):
        import torch
        contract = self.summary_contract
        tensors = [] if self.memory is None else [self.memory]
        depths = list(self.memory_depths)
        positions = list(self.memory_positions)
        base_shift = len(active) - len(closed)
        shift = 0
        for entry in self._folded_branches:
            if entry["offset"] > len(closed):
                continue
            if self.consumed_ids is not None and entry["offset"] < len(self.consumed_ids):
                raise RuntimeError("unconsumed child checkpoint precedes parent compact frontier")
            at = entry["offset"] + base_shift + shift
            label = list(self.tokenizer.encode(
                "\n[Sub-agent execution checkpoint: " + entry["child_id"] + "]\n",
                add_special_tokens=False))
            before = (label + list(contract["student_summary_before_ids"]) + entry["summary_ids"]
                + list(contract["student_summary_after_ids"]) + list(contract["student_memory_before_ids"]))
            slots = int(entry["memory"].shape[0])
            block = before + [int(self.tokenizer.pad_token_id)] * slots + list(contract["student_memory_after_ids"])
            if positions and positions[-1] >= at:
                raise RuntimeError("child fold would reorder existing residual positions")
            positions.extend(range(at + len(before), at + len(before) + slots))
            tensors.append(entry["memory"])
            depths.extend(entry["depths"])
            active[at:at] = block
            shift += len(block)
        self.memory_positions = positions
        if getattr(self, "server_memory", False):
            key = tuple(t.descriptor["handle_id"] for t in tensors)
            if key != self._context_memory_key or self._context_memory is None:
                previous = self._context_memory
                self._context_memory = None if not tensors else self.client.compose_memory(tensors)
                # Only a temporary concatenation is owned here; main and child
                # frontiers remain valid until a successful compact consumes them.
                roots = [self.memory, *(entry["memory"] for entry in self._folded_branches)]
                if previous is not None and all(previous is not root for root in roots):
                    self._release_retired([previous])
                self._context_memory_key = key
        else:
            self._context_memory = (None if not tensors else tensors[0] if len(tensors) == 1 else torch.cat(tensors))
        self._context_depths = depths
        return active

    def _fold_source_identity(self):
        return [entry["fold_id"] for entry in self._folded_branches]

    def _consume_prepared_folds(self, prepared):
        consumed = set(prepared.get("fold_ids", ()))
        self._folded_branches = [entry for entry in self._folded_branches if entry["fold_id"] not in consumed]
