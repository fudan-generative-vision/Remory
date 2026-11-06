"""Residual client whose only memory state is server-owned descriptors."""
import logging
import threading
import uuid

from context_compression.qwen38_residual_sglang_client import SummaryResidualSGLangClient
from context_compression.remory_memory import CAPABILITY, ROUTE, RemoteMemory, transition_digest
from context_compression.sglang_compressor_hook import (
    CUSTOM_PARAM_KEY, GPU_MEMORY_HANDLE_META_KEY, GPU_MEMORY_HANDLE_PUBLISH_SCHEMA,
    compute_memory_handle_identity, build_recover_custom_params,
)


class ServerResidualSGLangClient(SummaryResidualSGLangClient):
    server_memory = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._owned_memory = {}
        self._owned_lock = threading.RLock()
        self._snapshots = {}
        self._retired_memory = {}
        self._retired_snapshots = set()
        self._preserved_snapshots = set()
        self._checkpoint_snapshots = set()
        self._lease_stop = threading.Event()
        self._lease_thread = None

    def attest_residual_server(self, checkpoint, server_info=None):
        info = self.server_info() if server_info is None else server_info
        if CAPABILITY not in info.get("request_capabilities", []):
            raise RuntimeError("endpoint lacks server-owned residual memory; pin a compatible backend")
        handle_info = info.get("specforge_gpu_memory_handle") or {}
        self._memory_epoch = handle_info["server_epoch"]
        self._memory_hook_digest = handle_info["hook_abi_digest"]
        receipt = super().attest_residual_server(checkpoint, info)
        receipt["transport"] = CAPABILITY
        receipt["memory_budget"] = None
        self._residual_attestation = receipt
        interval = min(60., float(handle_info["ttl_seconds"]) / 3)
        def renew():
            while not self._lease_stop.wait(interval):
                with self._owned_lock:
                    try:
                        for value in list(self._retired_memory.values()):
                            try:
                                self.release_memory(value)
                            except Exception:
                                logging.getLogger(__name__).exception("Retrying retired memory cleanup failed")
                        self._delete_retired_snapshots()
                        active = [v.descriptor for k, v in self._owned_memory.items() if k not in self._retired_memory]
                        if active:
                            self._control("renew", handles=active, lease=True)
                    except Exception:
                        logging.getLogger(__name__).exception("Memory lease renewal failed")
        self._lease_thread = threading.Thread(target=renew, name="remory-memory-lease", daemon=True)
        self._lease_thread.start()
        return dict(receipt)

    def owned_memory(self):
        with self._owned_lock:
            return dict(self._owned_memory)

    def _remember(self, descriptor, expected_rows=None):
        value = RemoteMemory(descriptor, self.base_url)
        # Take cleanup ownership before checking the expected shape.
        with self._owned_lock:
            self._owned_memory[descriptor["handle_id"]] = value
        if descriptor["server_epoch"] != self._memory_epoch:
            raise RuntimeError("server memory belongs to a different epoch")
        width = self._residual_config["compressor"]["target_hidden_size"]
        if value.shape[1] != width or (expected_rows is not None and value.shape[0] != expected_rows):
            self.release_memory(value)
            raise RuntimeError("server returned an invalid memory shape")
        return value

    def _check_memory(self, value):
        if not isinstance(value, RemoteMemory) or value.backend != self.base_url:
            raise ValueError("memory must belong to this pinned backend")
        if value.descriptor["server_epoch"] != self._memory_epoch:
            raise ValueError("stale memory epoch; restore the remote checkpoint first")
        return value.descriptor

    def _encode_residual(self, ids, *, start, end, operation, input_depth=0,
                         summary_features=None, child_memory=None, expected_rows,
                         source_memory_positions=None, block_input_depths=None):
        from context_compression.qwen38_residual_sglang_hook import (
            RESIDUAL_SGLANG_ABI, validate_residual_encode_payload,
        )
        self._validate_generate_lifecycle_locked()
        if self._residual_attestation is None or len(ids) > self._residual_context_limit:
            raise ValueError("unattested residual request or source exceeds context capacity")
        residual = dict(abi=RESIDUAL_SGLANG_ABI, operation=operation, input_depth=input_depth)
        if summary_features is not None:
            residual["summary_handle"] = self._check_memory(summary_features)
        if block_input_depths is not None:
            residual["block_input_depths"] = list(block_input_depths)
        config = dict(version=1, mode="encode", evidence_spans=[[start, end]],
                      source_token_lengths=[end-start], residual=residual)
        if child_memory is not None:
            config.update(memory_handle=self._check_memory(child_memory), memory_positions=(
                list(source_memory_positions) if source_memory_positions is not None else list(range(start, end))))
        request_digest = transition_digest(ids, config)
        identity = compute_memory_handle_identity(server_epoch=self._memory_epoch,
            request_digest=request_digest, hook_abi_digest=self._memory_hook_digest)
        config["memory_handle_publish"] = dict(schema=GPU_MEMORY_HANDLE_PUBLISH_SCHEMA,
            op_id="residual." + uuid.uuid4().hex, memory_identity=identity, request_digest=request_digest)
        validate_residual_encode_payload(config, token_count=len(ids),
            hidden_size=self._residual_config["compressor"]["target_hidden_size"],
            max_depth=self._residual_config["compressor"]["max_depth"])
        response = self._post_generate(dict(input_ids=list(ids), extra_key="residual-" + uuid.uuid4().hex,
            sampling_params=dict(temperature=0., max_new_tokens=0, custom_params={CUSTOM_PARAM_KEY: config}),
            return_hidden_states=False, custom_logit_processor=self._processor_string(), cache_policy="bypass"))
        meta = response.get("meta_info") or {}
        value = self._remember(meta[GPU_MEMORY_HANDLE_META_KEY], expected_rows)
        try:
            self._require_cache_bypass_receipt(response)
            if any(k in meta for k in ("hidden_states", "memory_embeddings")):
                raise RuntimeError("server memory response leaked tensor data")
            if value.descriptor["memory_identity"] != identity or value.descriptor["request_digest"] != request_digest:
                raise RuntimeError("server memory transition identity mismatch")
            return value
        except BaseException:
            self.release_memory(value)
            raise

    def compose_memory(self, values):
        if len(values) == 1:
            return values[0]
        return self._compose([(v, 0, v.shape[0]) for v in values])

    def _control(self, operation, *, lease=False, **kwargs):
        result = self._post(ROUTE + ("/lease" if lease else ""), {"payload": dict(operation=operation, **kwargs)},
                            request_timeout=10 if lease or operation == "delete_snapshot" else None)
        if not result.get("success"):
            raise RuntimeError("server memory operation failed: " + str(result.get("message")))
        return result["result"]

    def _compose(self, parts):
        result = self._control("compose", op_id="compose." + uuid.uuid4().hex,
            parts=[dict(handle=self._check_memory(v), start=a, stop=b) for v, a, b in parts])
        return self._remember(result["handle"], sum(b-a for _, a, b in parts))

    def save_memory(self, value, *, preserve=False):
        key = value.descriptor["handle_id"]
        if key not in self._snapshots:
            self._snapshots[key] = self._control("save", handle=self._check_memory(value))
        snapshot = self._snapshots[key]
        if preserve:
            self._preserved_snapshots.add(snapshot["snapshot"])
        return dict(snapshot)

    def load_memory(self, snapshot, *, preserve=False):
        if snapshot["hook_abi_digest"] != self._memory_hook_digest:
            raise ValueError("snapshot belongs to a different compressor")
        self._checkpoint_snapshots.add(snapshot["snapshot"])
        if preserve:
            self._preserved_snapshots.add(snapshot["snapshot"])
        result = self._control("load", snapshot=snapshot["snapshot"], op_id="load." + uuid.uuid4().hex)
        value = self._remember(result["handle"], snapshot["shape"][0])
        self._snapshots[value.descriptor["handle_id"]] = dict(snapshot)
        return value

    def import_memory(self, tensor):
        # Only for resuming a pre-refactor local checkpoint. Normal encode and
        # generate never use this operation or serialize memory values.
        result = self._control("import", op_id="import." + uuid.uuid4().hex,
                               values=tensor.detach().float().cpu().tolist())
        value = self._remember(result["handle"], tensor.shape[0])
        self.save_memory(value)
        return value

    def checkpoint_saved(self, snapshots):
        previous = self._checkpoint_snapshots
        self._checkpoint_snapshots = set(snapshots)
        # Called only after the client journal's atomic write succeeded.
        self._retired_snapshots.update(previous - self._checkpoint_snapshots - self._preserved_snapshots)
        self._delete_retired_snapshots()

    def _delete_retired_snapshots(self):
        for key in list(self._retired_snapshots):
            try:
                self._control("delete_snapshot", snapshot=key)
                self._retired_snapshots.discard(key)
            except Exception:
                logging.getLogger(__name__).exception("Retired checkpoint cleanup failed")

    def _slice_memory(self, value, start, stop):
        return self._compose([(value, start, stop)])

    def release_memory(self, value):
        if value is None:
            return
        self._check_memory(value)
        key = value.descriptor["handle_id"]
        with self._owned_lock:
            self._retired_memory[key] = value
            self.release_memory_handle(value.descriptor, operation_id="release." + uuid.uuid4().hex, request_timeout=10)
            snapshot = self._snapshots.get(key)
            if snapshot and snapshot["snapshot"] not in self._preserved_snapshots | self._checkpoint_snapshots:
                self._control("delete_snapshot", snapshot=snapshot["snapshot"])
            self._snapshots.pop(key, None)
            self._owned_memory.pop(key, None)
            self._retired_memory.pop(key, None)

    def build_residual_memory(self, *args, **kwargs):
        # A serving policy, deliberately independent of the frozen training budget.
        kwargs["memory_budget"] = None
        before = set(self.owned_memory())
        final = None
        try:
            result = super().build_residual_memory(*args, **kwargs)
            value = result["memory_embeddings"]
            result["receipt"]["remote_snapshot"] = self.save_memory(value)
            result["receipt"]["memory_backend"] = self.base_url
            final = value.descriptor["handle_id"]
            return result
        finally:
            # Summary features and partial candidates never accumulate. The old
            # main/child memories predate this transaction and remain untouched.
            owned = self.owned_memory()
            for key in set(owned) - before - {final}:
                try:
                    self.release_memory(owned[key])
                except Exception:
                    logging.getLogger(__name__).exception("Could not release intermediate memory %s", key)

    def generate_with_memory(self, prefix_input_ids, *, memory_positions, memory_embeddings,
                             max_new_tokens, expected_hidden_size=4096,
                             sampling_overrides=None, image_data=None):
        params = build_recover_custom_params(memory_positions=memory_positions,
            memory_handle=self._check_memory(memory_embeddings), hidden_size=expected_hidden_size)
        # Residual recovery retains its established cache-bypass semantics.
        return self._generate(list(prefix_input_ids), max_new_tokens=max_new_tokens,
            custom_params=params, sampling_overrides=sampling_overrides,
            memory_transport_cache_policy=None, image_data=image_data)

    def close_memory(self):
        self._lease_stop.set()
        if self._lease_thread is not None:
            self._lease_thread.join(timeout=1)
        # Keep the last committed snapshot for resume; intermediate/retired
        # generations were deleted during replacement, not accumulated here.
        for snapshot in self._snapshots.values():
            self._preserved_snapshots.add(snapshot["snapshot"])
        for value in list(self._owned_memory.values()):
            try:
                self.release_memory(value)
            except Exception:
                logging.getLogger(__name__).exception("Could not release session memory")
